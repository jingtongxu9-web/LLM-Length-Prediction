"""Build the ALPS+PLP Hybrid v3 design set and unopened holdout."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_length_prediction.prompt_manifest import (
    LENGTH_ORDER,
    LENGTH_RANGES,
    _code_prompts,
    _qa_prompts,
    _summary_prompts,
)

DATASET_VERSION = "alps-plp-hybrid-v3-2026"
SEEDS = [42, 43, 44]

NEW_QA_FAMILIES = [
    ("federated_learning_privacy", "联邦学习为什么不能自动保证隐私，它还需要哪些安全机制？"),
    ("carbon_pricing", "碳税和碳排放交易分别如何影响企业减排决策？"),
    ("crispr_off_target", "CRISPR 基因编辑为什么会出现脱靶效应，研究人员如何降低风险？"),
    ("raft_consensus", "Raft 共识算法如何通过领导者选举和日志复制保持一致性？"),
]

NEW_SUMMARY_FAMILIES = [
    (
        "battery_recycling",
        "动力电池回收网络",
        "某地区在新能源汽车保有量快速增长后建设动力电池回收网络。监管部门要求销售商记录电池编码，维修企业和回收网点上传流向，具备资质的工厂负责检测、梯次利用或材料再生。试运行中，正规渠道回收量上升，但小型维修点担心录入成本，部分车主也不了解废旧电池的安全风险。项目随后提供简化工具和回收补贴，并加强对非法拆解的检查。评估发现，梯次利用能延长部分电池寿命，但必须根据健康状态匹配低风险场景；严重损伤电池应直接进入材料回收。报告建议继续完善责任追踪、运输安全和再生材料质量标准。",
    ),
    (
        "wildfire_warning",
        "森林火灾预警试点",
        "一个山地林区部署了卫星热点、瞭望摄像头、气象站和巡护员上报相结合的火灾预警系统。干燥季节中，系统缩短了若干烟点的发现时间，但云层遮挡、工业热源和设备故障也产生误报。指挥中心随后按照风速、湿度、植被和附近居民点对告警分级，并保留人工复核。社区演练显示，预警信息只有与明确的疏散路线、通信备份和责任分工结合才有效。项目总结认为，更多传感器不必然带来更准确的判断，长期运行还需要设备维护、跨部门数据共享和对误报漏报的持续复盘。",
    ),
    (
        "school_meals",
        "学校营养餐改进",
        "某县对义务教育阶段营养餐计划进行改进。过去菜单统一但季节变化少，部分学校存在剩餐较多和配送温度不稳定的问题。教育、卫生和农业部门共同制定营养底线，允许学校根据本地食材调整菜单，并建立留样和温度记录。学生参与试吃后，蔬菜接受度有所提高，但偏远学校的冷链成本仍然较高。项目引入区域中央厨房与应急备用供应商，同时公开采购和抽检结果。评估强调，营养餐不能只看每餐热量，还要关注蛋白质和微量营养素、食品安全、学生实际摄入以及困难家庭在假期中的营养连续性。",
    ),
    (
        "water_reuse",
        "工业园区再生水利用",
        "一个缺水工业园区建设再生水系统，将污水处理厂出水进一步处理后用于冷却、清洗和绿化。首年数据显示，自来水使用量下降，但不同企业对水质指标的要求差异较大，管网切换时也出现短时波动。园区因此按照用途分级供水，增加在线监测和事故旁路，并要求关键生产环节保留备用水源。部分居民担心再生水影响地下水，管理方公开监测数据并设置独立评估。报告认为，再生水能够缓解用水压力，但节水收益必须与处理能耗、浓缩废水处置、管网维护和长期健康风险一起评价。",
    ),
]

NEW_CODE_FAMILIES = [
    ("interval_set", "实现一个支持插入区间、合并重叠区间并查询覆盖总长度的 IntervalSet。"),
    ("idempotency_store", "实现一个支持并发请求、过期时间和结果复用的幂等键存储组件。"),
    (
        "dependency_injection",
        "实现一个支持构造函数依赖、单例生命周期和循环依赖检测的轻量依赖注入容器。",
    ),
    ("streaming_quantile", "实现一个内存有界的流式分位数估计器，并支持合并两个估计器的状态。"),
]


def _new_families() -> dict[str, list[tuple[str, dict[str, str]]]]:
    return {
        "qa": [(slug, _qa_prompts(question)) for slug, question in NEW_QA_FAMILIES],
        "summarization": [
            (slug, _summary_prompts(title, source)) for slug, title, source in NEW_SUMMARY_FAMILIES
        ],
        "code": [(slug, _code_prompts(requirement)) for slug, requirement in NEW_CODE_FAMILIES],
    }


def build_hybrid_v3_records(
    v1_manifest: str | Path = "data/prompts/alps_v1_prompts.jsonl",
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(v1_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        record["dataset_version"] = DATASET_VERSION
        record["split"] = "train"
        record["provenance"] = "opened_v1_design_data"
    for task, families in _new_families().items():
        for index, (slug, prompts) in enumerate(families, 1):
            family_id = f"hybrid_v3_{task}_{index:03d}_{slug}"
            for intended_length in LENGTH_ORDER:
                records.append(
                    {
                        "dataset_version": DATASET_VERSION,
                        "prompt_family_id": family_id,
                        "prompt_id": f"{family_id}_{intended_length}",
                        "task_type": task,
                        "intended_length": intended_length,
                        "intended_output_tokens": LENGTH_RANGES[intended_length],
                        "language": "zh-CN",
                        "split": "test",
                        "generation_seeds": SEEDS,
                        "prompt": prompts[intended_length],
                        "provenance": "new_unopened_hybrid_v3_holdout",
                    }
                )
    validate_hybrid_v3_records(records)
    return records


def validate_hybrid_v3_records(records: list[dict[str, Any]]) -> None:
    if len(records) != 216:
        raise ValueError(f"expected 216 prompts, found {len(records)}")
    if len({record["prompt_id"] for record in records}) != len(records):
        raise ValueError("prompt_id values must be unique")
    if Counter(record["split"] for record in records) != {"train": 180, "test": 36}:
        raise ValueError("expected 180 Train and 36 Test prompts")
    expected = {
        ("train", "qa"): 60,
        ("train", "summarization"): 60,
        ("train", "code"): 60,
        ("test", "qa"): 12,
        ("test", "summarization"): 12,
        ("test", "code"): 12,
    }
    if Counter((row["split"], row["task_type"]) for row in records) != expected:
        raise ValueError("task/split counts are not balanced")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[record["prompt_family_id"]].append(record)
    if len(families) != 72:
        raise ValueError("expected 72 prompt families")
    for family_id, rows in families.items():
        if len(rows) != 3 or len({row["split"] for row in rows}) != 1:
            raise ValueError(f"family {family_id} is incomplete or crosses splits")
        if {row["intended_length"] for row in rows} != set(LENGTH_ORDER):
            raise ValueError(f"family {family_id} is missing a length variant")
        if any(row["generation_seeds"] != SEEDS for row in rows):
            raise ValueError(f"family {family_id} does not use frozen seeds")


def write_hybrid_v3_manifest(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output
