"""可复用 DAG 模板模块的端到端测试。

覆盖：模板创建（含图校验复用与重名 409）、按名称发布不可变新版本、
版本/模板查询、payload 占位符参数提取与替换、缺参/多参 422 且不落库、
idempotency_key 幂等（同键同参返回同一 DAG、参数不同 409）、校验/替换/
DAG 创建/幂等记录单事务回滚、已发布快照存储层不可变、重启持久化，
以及直接 POST /dags 与既有接口行为不变。
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "templates.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("DEFAULT_LEASE_SECONDS", "30")

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def create_template(client, name, tasks, status=201):
    return client.post("/templates", json={"name": name, "tasks": tasks})


def publish(client, name, tasks):
    return client.post(f"/templates/{name}/versions", json={"tasks": tasks})


def instantiate(client, name, version, key, parameters=None):
    body = {"idempotency_key": key}
    if parameters is not None:
        body["parameters"] = parameters
    return client.post(
        f"/templates/{name}/versions/{version}/instantiate", json=body
    )


ETL_TASKS = [
    {"key": "extract", "payload": {"table": "${table}", "limit": 100}},
    {"key": "transform", "deps": ["extract"], "payload": "src=${table};host=${host}"},
    {"key": "load", "deps": ["transform"], "payload": {"targets": ["${host}"]}},
]


# ------------------------------------------------------------- 创建

def test_create_template_publishes_version_1_with_snapshot(client):
    r = create_template(client, "etl", ETL_TASKS)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "etl"
    assert body["latest_version"] == 1 and body["version_count"] == 1
    v1 = body["version"]
    assert v1["version"] == 1
    assert v1["task_count"] == 3
    # 占位符从全部任务 payload 提取，去重后排序
    assert v1["parameters"] == ["host", "table"]
    # 任务图快照保留占位符原文
    assert v1["tasks"][0]["payload"]["table"] == "${table}"
    assert v1["created_at"]


def test_create_template_duplicate_name_returns_409(client):
    assert create_template(client, "etl", [{"key": "a"}]).status_code == 201
    r = create_template(client, "etl", [{"key": "b"}])
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "template_exists"
    # 首次发布不受影响
    got = client.get("/templates/etl").json()
    assert got["version_count"] == 1


def test_create_template_reuses_dag_validation(client):
    # 缺失依赖
    r = create_template(client, "t-missing", [{"key": "a", "deps": ["ghost"]}])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "missing_dependency"
    # 环
    r = create_template(
        client,
        "t-cycle",
        [{"key": "a", "deps": ["c"]}, {"key": "b", "deps": ["a"]}, {"key": "c", "deps": ["b"]}],
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "cycle_detected"
    # 重复任务键
    r = create_template(client, "t-dup", [{"key": "a"}, {"key": "a"}])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "duplicate_task_key"
    # pydantic 层：空任务列表 / 缺名称
    assert client.post("/templates", json={"name": "x", "tasks": []}).status_code == 422
    assert client.post("/templates", json={"tasks": [{"key": "a"}]}).status_code == 422
    # 失败的创建不留模板
    assert client.get("/templates").json()["count"] == 0


def test_create_template_validates_concurrency_group_pair(client):
    # 同直接提交口径：组名与上限必须同时给出
    r = create_template(client, "t-g", [{"key": "a", "concurrency_group": "g"}])
    assert r.status_code == 422
    # 同批自相矛盾 422
    r = create_template(
        client,
        "t-g2",
        [
            {"key": "a", "concurrency_group": "g", "concurrency_limit": 1},
            {"key": "b", "concurrency_group": "g", "concurrency_limit": 2},
        ],
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "concurrency_group_conflict"


# ------------------------------------------------------------- 发布新版本

def test_publish_version_creates_immutable_incrementing_snapshots(client):
    create_template(client, "etl", ETL_TASKS)
    r = publish(client, "etl", [{"key": "only", "payload": {"q": "${q}"}}])
    assert r.status_code == 201, r.text
    v2 = r.json()
    assert v2["version"] == 2 and v2["parameters"] == ["q"] and v2["task_count"] == 1

    # v1 快照保持原样：新版本不修改历史版本
    v1 = client.get("/templates/etl/versions/1").json()
    assert v1["parameters"] == ["host", "table"] and v1["task_count"] == 3
    assert [t["key"] for t in v1["tasks"]] == ["extract", "transform", "load"]

    # 再发 v3
    r = publish(client, "etl", [{"key": "a"}, {"key": "b", "deps": ["a"]}])
    assert r.status_code == 201 and r.json()["version"] == 3
    got = client.get("/templates/etl").json()
    assert got["latest_version"] == 3 and got["version_count"] == 3


def test_publish_version_unknown_template_returns_404(client):
    r = publish(client, "ghost", [{"key": "a"}])
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "template_not_found"


def test_publish_version_validates_graph_and_leaves_no_version(client):
    create_template(client, "etl", ETL_TASKS)
    r = publish(client, "etl", [{"key": "a", "deps": ["a"]}])
    assert r.status_code == 422 and r.json()["error"]["code"] == "cycle_detected"
    # 失败发布不占版本号：下一次成功发布仍是 v2
    r = publish(client, "etl", [{"key": "a"}])
    assert r.status_code == 201 and r.json()["version"] == 2


def test_published_version_snapshot_is_immutable_in_storage(client):
    create_template(client, "etl", ETL_TASKS)
    from app.db import db_path

    conn = sqlite3.connect(db_path())
    with pytest.raises(sqlite3.Error):
        conn.execute("UPDATE dag_template_versions SET tasks='[]' WHERE name='etl'")
    conn.rollback()
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM dag_template_versions WHERE name='etl'")
    conn.rollback()
    conn.close()
    # 触发器阻止后快照仍可读且未变
    v1 = client.get("/templates/etl/versions/1").json()
    assert v1["task_count"] == 3


# ------------------------------------------------------------- 查询

def test_list_and_get_templates(client):
    create_template(client, "etl", ETL_TASKS)
    publish(client, "etl", [{"key": "a"}])
    create_template(client, "other", [{"key": "z"}])

    listed = client.get("/templates").json()
    assert listed["count"] == 2
    by_name = {t["name"]: t for t in listed["templates"]}
    assert by_name["etl"]["latest_version"] == 2 and by_name["etl"]["version_count"] == 2
    assert by_name["other"]["latest_version"] == 1
    # updated_at 反映最近一次发布
    assert by_name["etl"]["updated_at"] >= by_name["etl"]["created_at"]

    detail = client.get("/templates/etl").json()
    assert [v["version"] for v in detail["versions"]] == [1, 2]
    # 列表摘要不含快照正文
    assert "tasks" not in detail["versions"][0]
    assert detail["versions"][0]["parameters"] == ["host", "table"]


def test_get_version_404_codes(client):
    create_template(client, "etl", ETL_TASKS)
    assert client.get("/templates/ghost").status_code == 404
    r = client.get("/templates/etl/versions/99")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "template_version_not_found"
    r = client.get("/templates/ghost/versions/1")
    assert r.status_code == 404 and r.json()["error"]["code"] == "template_not_found"


# ------------------------------------------------------------- 实例化：替换

def test_instantiate_replaces_payload_placeholders(client):
    create_template(client, "etl", ETL_TASKS)
    r = instantiate(client, "etl", 1, "run-1", {"table": "users", "host": "db1"})
    assert r.status_code == 201, r.text
    dag = r.json()
    assert dag["name"] == "etl" and dag["task_count"] == 3
    tasks = {t["key"]: t for t in dag["tasks"]}
    assert tasks["extract"]["payload"] == {"table": "users", "limit": 100}
    assert tasks["transform"]["payload"] == "src=users;host=db1"
    assert tasks["load"]["payload"] == {"targets": ["db1"]}
    # key/deps/name 不参与替换
    assert set(tasks) == {"extract", "transform", "load"}
    assert tasks["load"]["deps"] == ["transform"]


def test_instantiate_dag_uses_existing_response_shape_and_is_claimable(client):
    create_template(client, "etl", ETL_TASKS)
    dag = instantiate(client, "etl", 1, "run-1", {"table": "users", "host": "db1"}).json()
    # 与 GET /dags/{id} 同一响应结构
    assert client.get(f"/dags/{dag['id']}").json() == dag
    assert dag["states"]["pending"] == 3

    # 实例化出的 DAG 与直接提交完全同构：可领取、依赖门控生效
    claimed = client.post("/tasks/claim", json={"dag_id": dag["id"], "limit": 10}).json()["claimed"]
    assert [t["task_key"] for t in claimed] == ["extract"]
    assert claimed[0]["payload"] == {"table": "users", "limit": 100}


def test_instantiate_each_version_is_independent_snapshot(client):
    create_template(client, "etl", [{"key": "a", "payload": "${x}"}])
    publish(client, "etl", [{"key": "a", "payload": {"x": "${x}", "y": "${y}"}}])

    d1 = instantiate(client, "etl", 1, "k1", {"x": "hello"}).json()
    d2 = instantiate(client, "etl", 2, "k2", {"x": "1", "y": "2"}).json()
    assert d1["id"] != d2["id"]
    assert d1["tasks"][0]["payload"] == "hello"
    assert d2["tasks"][0]["payload"] == {"x": "1", "y": "2"}


def test_instantiate_non_string_parameters(client):
    # 字符串原样内插；其他 JSON 类型按 JSON 文本内插
    create_template(client, "typed", [
        {"key": "a", "payload": {"n": "${n}", "b": "${b}", "null": "${z}", "arr": "${arr}"}},
    ])
    params = {"n": 42, "b": True, "z": None, "arr": [1, "two", False]}
    r = instantiate(client, "typed", 1, "k", params)
    assert r.status_code == 201, r.text
    payload = r.json()["tasks"][0]["payload"]
    assert payload == {"n": "42", "b": "true", "null": "null", "arr": '[1,"two",false]'}


def test_instantiate_template_without_parameters(client):
    create_template(client, "plain", [{"key": "a", "payload": {"fixed": True}}])
    r = instantiate(client, "plain", 1, "k")
    assert r.status_code == 201
    assert r.json()["tasks"][0]["payload"] == {"fixed": True}
    # 无参模板传任何参数都算多参
    r = instantiate(client, "plain", 1, "k2", {"x": 1})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unexpected_template_parameter"


def test_instantiate_nested_placeholders(client):
    create_template(client, "nested", [
        {"key": "a", "payload": {"outer": [{"inner": "${p}"}, "${p}", 1, None]}},
    ])
    r = instantiate(client, "nested", 1, "k", {"p": "v"})
    assert r.status_code == 201
    assert r.json()["tasks"][0]["payload"] == {"outer": [{"inner": "v"}, "v", 1, None]}


# ----------------------------------------------------- 缺参 / 多参 / 校验

def test_instantiate_missing_parameter_returns_422(client):
    create_template(client, "etl", ETL_TASKS)
    r = instantiate(client, "etl", 1, "run-x", {"table": "users"})  # 缺 host
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "missing_template_parameter"
    assert err["details"]["missing"] == ["host"]


def test_instantiate_unexpected_parameter_returns_422(client):
    create_template(client, "etl", ETL_TASKS)
    r = instantiate(client, "etl", 1, "run-x",
                    {"table": "users", "host": "db1", "extra": 1})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unexpected_template_parameter"
    assert err["details"]["unexpected"] == ["extra"]


def test_instantiate_validation_failure_persists_nothing(client):
    """缺参/多参 422 不落库：无 DAG、无事件、无幂等记录，同键可立即重试。"""
    create_template(client, "etl", ETL_TASKS)
    assert instantiate(client, "etl", 1, "run-1", {"table": "users"}).status_code == 422
    assert instantiate(client, "etl", 1, "run-1",
                       {"table": "u", "host": "h", "x": 1}).status_code == 422

    # 同键以正确参数重试 -> 首次创建 201（失败未占用幂等键）
    r = instantiate(client, "etl", 1, "run-1", {"table": "users", "host": "db1"})
    assert r.status_code == 201
    dag = r.json()

    # 只产生了这一个 DAG 与一条 dag.created 事件
    assert client.get("/tasks", params={"dag_id": dag["id"]}).json()["count"] == 3
    events = client.get("/events", params={"dag_id": dag["id"]}).json()["events"]
    assert [e["type"] for e in events] == ["dag.created"]

    from app.db import db_path
    conn = sqlite3.connect(db_path())
    assert conn.execute("SELECT COUNT(*) FROM dags").fetchone()[0] == 1
    rows = conn.execute(
        "SELECT dag_id FROM dag_template_instantiations"
    ).fetchall()
    assert rows == [(dag["id"],)]
    conn.close()


def test_instantiate_idempotency_key_required(client):
    create_template(client, "etl", ETL_TASKS)
    r = client.post("/templates/etl/versions/1/instantiate",
                    json={"parameters": {"table": "u", "host": "h"}})
    assert r.status_code == 422
    r = client.post("/templates/etl/versions/1/instantiate",
                    json={"idempotency_key": "", "parameters": {"table": "u", "host": "h"}})
    assert r.status_code == 422


def test_instantiate_unknown_template_or_version_404(client):
    create_template(client, "etl", ETL_TASKS)
    r = instantiate(client, "ghost", 1, "k", {})
    assert r.status_code == 404 and r.json()["error"]["code"] == "template_not_found"
    r = instantiate(client, "etl", 99, "k", {})
    assert r.status_code == 404 and r.json()["error"]["code"] == "template_version_not_found"
    # 非整数版本号由路径校验拒绝
    assert client.post("/templates/etl/versions/latest/instantiate", json={"idempotency_key": "k"}).status_code == 422


# ------------------------------------------------------------- 幂等

def test_instantiate_idempotent_replay_returns_same_dag(client):
    create_template(client, "etl", ETL_TASKS)
    params = {"table": "users", "host": "db1"}
    r1 = instantiate(client, "etl", 1, "run-1", params)
    assert r1.status_code == 201
    r2 = instantiate(client, "etl", 1, "run-1", params)
    assert r2.status_code == 200
    assert r2.json()["id"] == r1.json()["id"]
    # 第三次同样幂等；GET 接口一致
    r3 = instantiate(client, "etl", 1, "run-1", {"host": "db1", "table": "users"})
    assert r3.status_code == 200 and r3.json() == r1.json()
    # 只有一个 DAG 落库
    assert client.get("/tasks").json()["count"] == 3


def test_instantiate_same_key_different_parameters_returns_409(client):
    create_template(client, "etl", ETL_TASKS)
    r1 = instantiate(client, "etl", 1, "run-1", {"table": "users", "host": "db1"})
    assert r1.status_code == 201
    dag_id = r1.json()["id"]

    r2 = instantiate(client, "etl", 1, "run-1", {"table": "orders", "host": "db1"})
    assert r2.status_code == 409
    err = r2.json()["error"]
    assert err["code"] == "idempotency_conflict"
    assert err["details"] == {"name": "etl", "version": 1, "idempotency_key": "run-1"}

    # 冲突不新建 DAG；原 DAG 不受影响
    assert client.get("/tasks").json()["count"] == 3
    assert client.get(f"/dags/{dag_id}").status_code == 200
    # 正确参数仍可重放
    assert instantiate(client, "etl", 1, "run-1",
                       {"table": "users", "host": "db1"}).status_code == 200


def test_idempotency_scoped_per_template_version_and_key(client):
    create_template(client, "etl", [{"key": "a", "payload": "${x}"}])
    publish(client, "etl", [{"key": "a", "payload": "${y}"}])
    create_template(client, "etl2", [{"key": "a", "payload": "${x}"}])

    d_v1 = instantiate(client, "etl", 1, "k", {"x": "1"}).json()
    d_v2 = instantiate(client, "etl", 2, "k", {"y": "2"}).json()
    d_t2 = instantiate(client, "etl2", 1, "k", {"x": "3"}).json()
    assert len({d_v1["id"], d_v2["id"], d_t2["id"]}) == 3


def test_concurrent_instantiation_same_key_is_atomic(client):
    import threading

    create_template(client, "etl", ETL_TASKS)
    results = []
    errors = []

    def worker():
        try:
            r = instantiate(client, "etl", 1, "same-key", {"table": "u", "host": "h"})
            results.append((r.status_code, r.json()["id"]))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors
    assert len(results) == 8
    # 一个 201 + 七个 200，全部返回同一个 DAG
    assert sorted(code for code, _ in results) == [200] * 7 + [201]
    assert len({dag_id for _, dag_id in results}) == 1


# ----------------------------------------------------- 单事务（409 回滚）

def test_concurrency_group_conflict_rolls_back_dag_and_idempotency_record(client):
    # 直接提交先持久化组上限 2
    client.post("/dags", json={
        "id": "direct",
        "tasks": [{"key": "g", "concurrency_group": "grp", "concurrency_limit": 2}],
    })
    # 模板快照声明上限 1：实例化时跨 DAG 组冲突 -> 409
    create_template(client, "templ", [
        {"key": "a", "concurrency_group": "grp", "concurrency_limit": 1},
    ])
    r = instantiate(client, "templ", 1, "k", {})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "concurrency_group_conflict"

    from app.db import db_path
    conn = sqlite3.connect(db_path())
    # 只有直接提交的 DAG；幂等记录未写入；无额外 dag.created
    assert [r[0] for r in conn.execute("SELECT id FROM dags")] == ["direct"]
    assert conn.execute("SELECT COUNT(*) FROM dag_template_instantiations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='dag.created'"
    ).fetchone()[0] == 1
    conn.close()
    # 模板本身仍可查询、可再次发布
    assert client.get("/templates/templ/versions/1").status_code == 200


# ------------------------------------------------------------- 重启持久化

def test_templates_and_idempotency_records_survive_restart(tmp_path, monkeypatch):
    db_file = tmp_path / "persist.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("DEFAULT_LEASE_SECONDS", "30")

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c1:
        r = c1.post("/templates", json={"name": "etl", "tasks": ETL_TASKS})
        assert r.status_code == 201
        d = c1.post(
            "/templates/etl/versions/1/instantiate",
            json={"idempotency_key": "run-1", "parameters": {"table": "u", "host": "h"}},
        ).json()
        dag_id = d["id"]

    with TestClient(app) as c2:
        # 模板与不可变版本仍在
        v1 = c2.get("/templates/etl/versions/1").json()
        assert v1["parameters"] == ["host", "table"] and v1["task_count"] == 3
        # 幂等记录仍在：重放返回同一 DAG（200）
        r = c2.post(
            "/templates/etl/versions/1/instantiate",
            json={"idempotency_key": "run-1", "parameters": {"table": "u", "host": "h"}},
        )
        assert r.status_code == 200 and r.json()["id"] == dag_id
        # 参数不同仍 409
        r = c2.post(
            "/templates/etl/versions/1/instantiate",
            json={"idempotency_key": "run-1", "parameters": {"table": "x", "host": "h"}},
        )
        assert r.status_code == 409
        # 重启后仍可发布新版本
        r = c2.post("/templates/etl/versions", json={"tasks": [{"key": "a"}]})
        assert r.status_code == 201 and r.json()["version"] == 2


# ------------------------------------------------------------- 兼容性

def test_direct_post_dags_behavior_unchanged(client):
    # 直接 POST /dags 不经模板/参数/幂等逻辑：201、自定义 id、重复 id 409
    r = client.post("/dags", json={"id": "direct-1", "name": "d", "tasks": [{"key": "a"}]})
    assert r.status_code == 201 and r.json()["id"] == "direct-1"
    r = client.post("/dags", json={"id": "direct-1", "tasks": [{"key": "b"}]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "dag_exists"


def test_template_endpoints_in_openapi(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    assert "post" in paths["/templates"]
    assert "get" in paths["/templates"]
    assert "get" in paths["/templates/{name}"]
    assert "post" in paths["/templates/{name}/versions"]
    assert "get" in paths["/templates/{name}/versions/{version}"]
    inst = paths["/templates/{name}/versions/{version}/instantiate"]["post"]
    assert {"200", "201", "404", "409", "422"} <= set(inst["responses"])
    pub = paths["/templates/{name}/versions"]["post"]
    assert {"201", "404", "422"} <= set(pub["responses"])
