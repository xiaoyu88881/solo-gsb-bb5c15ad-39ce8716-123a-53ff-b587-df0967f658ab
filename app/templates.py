"""可复用 DAG 模板：模板创建、按名称发布不可变新版本、版本查询与参数化实例化。

模板（dag_templates）以名称为唯一业务键；每次发布在 dag_template_versions 中
固化一个不可变的任务图快照（与 POST /dags 提交体同构），并提取任务 payload
中出现的 ``${name}`` 占位符作为该版本的参数集合。

实例化时用调用方参数替换占位符，随后复用 service 的 DAG 校验与落库逻辑：
缺参/多参（422）、图校验失败、跨 DAG 并发组冲突（409）与幂等记录全部在
**同一个 SQLite 事务**内裁决，任何失败都整体回滚、不留 DAG 与幂等痕迹。
同 (模板, 版本, idempotency_key) 的重复调用返回首次创建的同一 DAG；
键相同但参数不同返回 409。直接 POST /dags 的行为不受本模块影响。
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
import uuid
from typing import Any

from . import service
from .db import get_conn

# 任务 payload 中的参数占位符：${name}；name 为非空且不含 '}' 的名称。
# 占位符只在 payload 的字符串**值**（含嵌套 list/dict 的字符串值）内识别，
# 任务 key/deps/name 等其他字段不做替换。
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


class APIError(service.APIError):
    """复用全局错误处理器：与 service.APIError 同结构。"""


def _scan_placeholders(value: Any, found: set[str]) -> None:
    """递归收集 payload 字符串值中出现的全部占位符名。"""
    if isinstance(value, str):
        found.update(_PLACEHOLDER_RE.findall(value))
    elif isinstance(value, list):
        for item in value:
            _scan_placeholders(item, found)
    elif isinstance(value, dict):
        for item in value.values():  # 仅替换值；键保持原样
            _scan_placeholders(item, found)


def _replace_placeholders(value: Any, parameters: dict[str, Any]) -> Any:
    """递归把 payload 字符串值中的 ${name} 替换为对应参数。

    参数本身是字符串时原样内插；其他 JSON 类型按 JSON 文本内插
    （数字、布尔、null、对象、数组均可作为参数）。仅在已校验参数集合
    与占位符集合完全一致后调用。
    """
    if isinstance(value, str):
        if "${" not in value:
            return value

        def _sub(match: re.Match) -> str:
            param = parameters[match.group(1)]
            return param if isinstance(param, str) else json.dumps(param, ensure_ascii=False, separators=(",", ":"))

        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, list):
        return [_replace_placeholders(item, parameters) for item in value]
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, parameters) for key, item in value.items()}
    return value


def _snapshot_parameters(tasks: list[dict[str, Any]]) -> list[str]:
    """提取版本快照的参数集合：全部任务 payload 中占位符名去重后排序。"""
    found: set[str] = set()
    for task in tasks:
        _scan_placeholders(task.get("payload"), found)
    return sorted(found)


def _version_dict(row, include_tasks: bool) -> dict[str, Any]:
    parameters = json.loads(row["parameters"])
    body: dict[str, Any] = {
        "name": row["name"],
        "version": row["version"],
        "parameters": parameters,
        "task_count": row["task_count"],
        "created_at": row["created_at"],
    }
    if include_tasks:
        # 不可变任务图快照（发布时的规范化任务定义，含 payload 占位符原文）
        body["tasks"] = json.loads(row["tasks"])
    return body


# ---------------------------------------------------------------- 创建与发布

def create_template(body: dict[str, Any]) -> dict[str, Any]:
    """创建模板并发布不可变的第 1 版；重名返回 409 template_exists。

    与发布新版本走同一套快照校验（复用 DAG 图校验）；模板行与版本快照在
    单个事务内写入，失败整体回滚。
    """
    name = body["name"]
    tasks = body["tasks"]
    # 快照校验与 POST /dags 同口径（重复键/缺失依赖/环/并发组自相矛盾均 422）
    service.validate_dag_payload(copy.deepcopy(tasks))

    now = service._iso(service._now())
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            try:
                conn.execute(
                    "INSERT INTO dag_templates (name, created_at, updated_at) VALUES (?, ?, ?)",
                    (name, now, now),
                )
            except sqlite3.IntegrityError:
                raise APIError(409, "template_exists", f"DAG 模板 {name} 已存在")
            conn.execute(
                """INSERT INTO dag_template_versions
                       (name, version, tasks, parameters, task_count, created_at)
                   VALUES (?, 1, ?, ?, ?, ?)""",
                (
                    name,
                    json.dumps(tasks, ensure_ascii=False),
                    json.dumps(_snapshot_parameters(tasks), ensure_ascii=False),
                    len(tasks),
                    now,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        row = conn.execute(
            "SELECT * FROM dag_template_versions WHERE name = ? AND version = 1", (name,)
        ).fetchone()
        return {
            "name": name,
            "latest_version": 1,
            "version_count": 1,
            "created_at": now,
            "updated_at": now,
            "version": _version_dict(row, include_tasks=True),
        }


def publish_template_version(name: str, body: dict[str, Any]) -> dict[str, Any]:
    """按名称为已有模板发布下一个不可变版本；模板不存在返回 404 template_not_found。

    版本号在保留事务内取 max(version)+1，并发发布串行化、不重号。
    新版本可携带与旧版本不同的任务图与参数集合；旧版本快照永不改变。
    """
    tasks = body["tasks"]
    service.validate_dag_payload(copy.deepcopy(tasks))

    now = service._iso(service._now())
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            template = conn.execute(
                "SELECT * FROM dag_templates WHERE name = ?", (name,)
            ).fetchone()
            if template is None:
                raise APIError(404, "template_not_found", f"DAG 模板 {name} 不存在")
            latest = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM dag_template_versions WHERE name = ?",
                (name,),
            ).fetchone()["v"]
            version = int(latest) + 1
            conn.execute(
                """INSERT INTO dag_template_versions
                       (name, version, tasks, parameters, task_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    version,
                    json.dumps(tasks, ensure_ascii=False),
                    json.dumps(_snapshot_parameters(tasks), ensure_ascii=False),
                    len(tasks),
                    now,
                ),
            )
            conn.execute(
                "UPDATE dag_templates SET updated_at = ? WHERE name = ?", (now, name)
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        row = conn.execute(
            "SELECT * FROM dag_template_versions WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        return _version_dict(row, include_tasks=True)


# ---------------------------------------------------------------- 查询

def list_templates() -> dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.name, t.created_at, t.updated_at,
                      COUNT(v.version) AS version_count,
                      MAX(v.version)   AS latest_version
                 FROM dag_templates t
                 LEFT JOIN dag_template_versions v ON v.name = t.name
                GROUP BY t.name
                ORDER BY t.created_at, t.name"""
        ).fetchall()
        items = [
            {
                "name": r["name"],
                "latest_version": r["latest_version"],
                "version_count": r["version_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]
        return {"templates": items, "count": len(items)}


def get_template(name: str) -> dict[str, Any]:
    """模板详情：含全部版本（版本列表不含快照正文，需具体版本取 tasks）。"""
    with get_conn() as conn:
        template = conn.execute(
            "SELECT * FROM dag_templates WHERE name = ?", (name,)
        ).fetchone()
        if template is None:
            raise APIError(404, "template_not_found", f"DAG 模板 {name} 不存在")
        version_rows = conn.execute(
            "SELECT * FROM dag_template_versions WHERE name = ? ORDER BY version", (name,)
        ).fetchall()
        return {
            "name": template["name"],
            "latest_version": max(r["version"] for r in version_rows),
            "version_count": len(version_rows),
            "created_at": template["created_at"],
            "updated_at": template["updated_at"],
            "versions": [_version_dict(r, include_tasks=False) for r in version_rows],
        }


def get_template_version(name: str, version: int) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM dag_template_versions WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is None:
            # 模板本身不存在与版本号不存在同样 404，错误码区分便于调用方定位
            exists = conn.execute(
                "SELECT 1 FROM dag_templates WHERE name = ?", (name,)
            ).fetchone()
            if exists is None:
                raise APIError(404, "template_not_found", f"DAG 模板 {name} 不存在")
            raise APIError(
                404,
                "template_version_not_found",
                f"DAG 模板 {name} 的版本 {version} 不存在",
                {"name": name, "version": version},
            )
        return _version_dict(row, include_tasks=True)


# ---------------------------------------------------------------- 实例化

def instantiate_template(name: str, version: int, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """用参数实例化指定模板版本，返回 (HTTP 状态码, DAG 视图)。

    首次实例化返回 201；同 (模板, 版本, idempotency_key) 重复且参数等价时
    返回 200 与首次创建的同一 DAG。参数集合与版本占位符必须**恰好**一致：
    缺参或多参一律 422；同键参数不同返回 409。校验、占位符替换、DAG 创建
    与幂等记录在单个 BEGIN IMMEDIATE 事务内完成，失败不留任何痕迹。
    """
    parameters = body.get("parameters") or {}
    idempotency_key = body["idempotency_key"]
    fingerprint = service._canonical(parameters)
    now_iso = service._iso(service._now())

    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version_row = conn.execute(
                "SELECT * FROM dag_template_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            if version_row is None:
                exists = conn.execute(
                    "SELECT 1 FROM dag_templates WHERE name = ?", (name,)
                ).fetchone()
                if exists is None:
                    raise APIError(404, "template_not_found", f"DAG 模板 {name} 不存在")
                raise APIError(
                    404,
                    "template_version_not_found",
                    f"DAG 模板 {name} 的版本 {version} 不存在",
                    {"name": name, "version": version},
                )

            # 幂等记录先于参数替换与 DAG 创建检查：同键重放不重复建图
            existing = conn.execute(
                """SELECT * FROM dag_template_instantiations
                   WHERE name = ? AND version = ? AND idempotency_key = ?""",
                (name, version, idempotency_key),
            ).fetchone()
            if existing is not None:
                conn.execute("ROLLBACK")
                if existing["parameters"] != fingerprint:
                    raise APIError(
                        409,
                        "idempotency_conflict",
                        "相同模板版本与 idempotency_key 已用于参数不同的实例化",
                        {"name": name, "version": version, "idempotency_key": idempotency_key},
                    )
                return 200, service.get_dag(existing["dag_id"])

            required = json.loads(version_row["parameters"])
            provided = set(parameters)
            missing = [p for p in required if p not in provided]
            extra = sorted(provided - set(required))
            if missing:
                raise APIError(
                    422,
                    "missing_template_parameter",
                    "实例化参数缺少模板占位符要求的键",
                    {"name": name, "version": version, "missing": missing},
                )
            if extra:
                raise APIError(
                    422,
                    "unexpected_template_parameter",
                    "实例化参数包含模板未使用的多余键",
                    {"name": name, "version": version, "unexpected": extra},
                )

            # 从不可变快照复制任务定义并替换 payload 占位符；复用同一套 DAG
            # 校验（替换后再次校验，与 POST /dags 同口径）与落库逻辑
            tasks = copy.deepcopy(json.loads(version_row["tasks"]))
            for task in tasks:
                if task.get("payload") is not None:
                    task["payload"] = _replace_placeholders(task["payload"], parameters)
            keys, deps_map, tasks_by_key, declared = service.validate_dag_payload(tasks)

            dag_id = uuid.uuid4().hex
            service.persist_dag(conn, dag_id, name, now_iso, keys, deps_map, tasks_by_key, declared)
            conn.execute(
                """INSERT INTO dag_template_instantiations
                       (name, version, idempotency_key, parameters, dag_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, version, idempotency_key, fingerprint, dag_id, now_iso),
            )
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    return 201, service.get_dag(dag_id)
