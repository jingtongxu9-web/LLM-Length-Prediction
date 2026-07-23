"""Build the frozen ALPS v1 prompt manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATASET_VERSION = "alps-v1"
SPLIT_SEED = 42
LENGTH_ORDER = ("short", "medium", "long")
LENGTH_RANGES = {
    "short": {"min": 20, "max": 150},
    "medium": {"min": 150, "max": 500},
    "long": {"min": 500, "max": 1500},
}

QA_FAMILIES = [
    ("gradient_descent", "什么是梯度下降，它如何帮助机器学习模型找到较优参数？"),
    ("transformer_attention", "Transformer 的自注意力机制是怎样工作的？"),
    ("tcp_handshake", "TCP 为什么需要三次握手，而不是两次握手？"),
    ("inflation", "通货膨胀是如何形成的，它会怎样影响普通家庭？"),
    ("photosynthesis", "植物如何通过光合作用把光能转化为化学能？"),
    ("blockchain_consensus", "区块链共识机制解决了什么问题？"),
    ("database_index", "数据库索引为什么能加速查询，又会带来哪些成本？"),
    ("container_vs_vm", "容器与虚拟机在隔离方式和资源开销上有什么区别？"),
    ("bayes_theorem", "贝叶斯定理如何根据新证据更新判断？"),
    ("climate_feedback", "气候系统中的正反馈和负反馈分别意味着什么？"),
    ("public_key_crypto", "公钥密码学如何在不共享私钥的情况下实现安全通信？"),
    ("cache_consistency", "分布式系统为什么会出现缓存一致性问题？"),
    ("reinforcement_learning", "强化学习中的奖励、策略和价值函数有什么关系？"),
    ("supply_demand", "供给与需求如何共同影响市场价格？"),
    ("quantum_entanglement", "量子纠缠是什么，它为什么不等于超光速通信？"),
    ("dns_resolution", "浏览器访问一个域名时，DNS 解析经历了哪些步骤？"),
    ("immune_memory", "免疫系统为什么能在再次遇到病原体时更快响应？"),
    ("compiler_pipeline", "编译器如何把高级语言程序转换成可执行代码？"),
    ("electric_grid", "电力系统为什么必须实时维持发电与用电平衡？"),
    ("plate_tectonics", "板块构造运动如何导致地震、火山和山脉形成？"),
]

SUMMARY_FAMILIES = [
    (
        "urban_transit",
        "城市公共交通改革",
        "某沿海城市在三年内推进公共交通改革。改革前，高峰期公交平均时速只有每小时十四公里，换乘信息分散，郊区居民到中心城区通常需要两次以上换乘。市政府先整合公交、地铁和共享单车的数据平台，又调整重复线路，将部分运力转向早晚高峰。试点一年后，核心线路平均候车时间下降约百分之十八，但偏远地区因线路合并出现步行距离增加的问题。第二阶段增加了预约微循环巴士，并为老年人保留电话预约渠道。评估报告认为，改革提升了整体效率，却也提醒不能只看平均通勤时间，还要持续监测低收入社区、夜班劳动者和行动不便人群的可达性。",
    ),
    (
        "hybrid_work",
        "企业混合办公试点",
        "一家拥有两千名员工的软件企业进行了十八个月的混合办公试点。研发团队每周到办公室两天，客户支持和财务岗位则根据业务高峰安排现场值班。员工调查显示，通勤时间减少后满意度上升，但新人融入速度和跨团队信息共享有所下降。公司随后建立固定的协作日、导师制度和异步决策记录，并要求关键会议同时提供文字纪要。试点后期，项目交付周期没有明显变化，员工离职率略有下降，不过部分管理者反映协调成本转移到了会议安排和文档维护。最终报告建议根据岗位性质而不是统一比例制定规则，并把产出质量、知识流动和员工健康一起纳入评估。",
    ),
    (
        "wetland_restoration",
        "河口湿地修复项目",
        "某河口湿地长期受到围垦、养殖排水和外来植物扩张影响，候鸟停歇面积持续缩小。修复项目先拆除部分废弃堤坝，使潮水重新进入滩涂，再通过人工割除和水位调节控制外来植物。两年监测显示，本地盐沼植物覆盖率回升，春季记录到的候鸟种类增加，但鱼类幼体数量在不同季节波动明显。附近养殖户担心盐水倒灌，项目组因此增设水质监测点和应急闸门，并公开每月数据。专家指出，湿地恢复不是一次性绿化工程，需要长期协调防洪、生物多样性和社区生计，过早用单一物种数量宣布成功可能造成误判。",
    ),
    (
        "community_eldercare",
        "社区养老服务网络",
        "一个老龄化城区尝试建立十五分钟社区养老服务网络。街道将闲置物业改造成日间照料中心，提供助餐、康复训练、用药提醒和短时托养，同时培训社区志愿者定期探访独居老人。运行半年后，助餐服务使用率最高，数字健康设备的使用率却低于预期，原因包括操作复杂、隐私担忧和子女不在本地。项目随后简化设备界面，并允许老人选择纯人工服务。财政评估发现，集中照护能降低部分家庭的临时看护压力，但重度失能老人的专业护理仍然不足。报告建议把社区服务与医院、长期护理机构和紧急救援系统连接起来，并建立稳定的护理人员培训和薪酬机制。",
    ),
    (
        "smart_factory",
        "制造企业数字化改造",
        "一家中型零部件工厂计划用传感器和生产管理系统改造三条生产线。第一阶段采集设备振动、温度和停机原因，用于预测性维护；第二阶段把订单、库存和质检数据接入统一看板。实施初期，设备故障停机时间下降，但不同供应商的数据格式不一致，现场员工也担心系统被用于单纯考核个人。管理层随后成立联合小组，让操作员参与告警阈值和流程设计，并把改进目标从个人速度调整为整线稳定性。项目最终减少了部分紧急维修和在制品库存，不过软件维护、传感器校准和网络安全支出高于最初预算。总结认为，数字化收益来自流程重构，而不是简单安装更多设备。",
    ),
    (
        "student_wellbeing",
        "校园心理健康支持",
        "某大学针对学生心理压力上升建立了分层支持体系。第一层是面向所有学生的心理课程和匿名自测，第二层由辅导员和受训同伴提供早期识别，第三层则转介到专业咨询和医院。实施后，主动预约咨询的人数增加，危机个案的平均响应时间缩短，但咨询中心出现排队延长。学校增加团体辅导和夜间热线，同时明确自测结果不能替代诊断。学生代表提出，学业评价、经济压力和住宿环境也是重要来源，不能把问题全部归结为个人适应能力。项目报告建议继续扩充专业人员，并同步改善课程安排、困难资助和隐私保护。",
    ),
    (
        "cold_chain",
        "农产品冷链升级",
        "某农业县为减少蔬果采后损耗建设县域冷链网络。项目包括产地预冷库、冷藏运输车辆和面向批发商的温度追踪平台。首个产季中，叶菜和浆果的腐损率明显下降，农户可销售时间延长，但小规模种植户因最低起运量限制难以单独使用服务。合作社随后推出拼车运输和按箱计费，并安排技术员指导预冷流程。运营数据表明，冷库利用率在旺季接近上限，淡季却偏低，能源成本成为持续压力。评估建议将水产、乳制品等不同品类错峰接入，同时公开收费规则，避免基础设施只服务少数大型经营者。",
    ),
    (
        "open_source_governance",
        "开源项目治理调整",
        "一个快速增长的开源项目过去主要由三名核心维护者决定路线。随着企业用户和外部贡献者增加，代码审查积压、版本发布时间不稳定，社区还对商业功能优先级产生争议。项目随后成立技术委员会，公开提案流程，并为安全修复设置快速通道。治理调整后，普通贡献者更容易了解决策理由，但会议数量增加，部分紧急决策变慢。社区进一步引入任期轮换、利益冲突披露和异步投票，同时保留维护者对发布质量的最终责任。年度复盘认为，透明度提高并不自动等于高效率，成熟治理需要在参与广度、技术一致性和响应速度之间持续权衡。",
    ),
    (
        "data_center_energy",
        "数据中心节能计划",
        "某数据中心园区启动节能计划，目标是在不降低服务等级的前提下减少电力和冷却消耗。运维团队先提高部分机房送风温度，再根据服务器负载动态调整风机和冷水机。试验期间总体能耗下降，但局部机柜出现热点，说明平均温度不能代表全部风险。团队增加机柜级传感器，并把工作负载迁移与冷却控制联动。随后又在低峰期安排部分批处理任务，以利用夜间较低的环境温度。报告指出，节能效果必须同时考虑计算利用率、设备寿命和备用容量，不能通过延迟关键任务或减少安全冗余来换取表面指标。",
    ),
    (
        "sponge_city",
        "海绵城市街区改造",
        "一个老城区在频繁内涝后实施海绵化改造。工程包括透水铺装、下沉绿地、屋顶雨水收集和排水管网清淤。一次中等强度暴雨中，试点街区积水消退速度明显快于相邻区域，但极端降雨仍超过设施设计能力。居民认可绿地改善，却反映部分透水路面维护不及时，堵塞后效果下降。建设部门因此把年度清理、渗透率检测和社区报修纳入长期预算。评估强调，海绵设施只能削减和延迟径流，不能完全替代地下排水、防洪通道和应急管理，应根据不同降雨情景组合使用。",
    ),
    (
        "vocational_training",
        "职业教育校企合作",
        "一所职业院校与本地制造企业共同设计智能设备维护课程。企业提供真实设备和实习岗位，教师负责把现场任务转化为基础理论与安全训练。首届学生的实操考核成绩提高，但部分实习内容过度集中在重复操作，未能覆盖故障诊断和系统思维。学校随后建立实习任务清单，并要求企业导师与校内教师共同评分。企业认为学生上岗适应时间缩短，学生则希望技能证书能被更多行业认可。项目总结提出，校企合作不能只解决短期用工，还要保障通用能力、劳动权益和继续学习空间。",
    ),
    (
        "primary_care",
        "基层医疗联合门诊",
        "某地区为缓解大医院普通门诊压力，建立基层医疗联合门诊。社区医生可以通过远程平台邀请专科医生会诊，患者在社区完成复诊、慢病监测和常用药续方。试运行后，高血压和糖尿病患者的随访完成率提高，前往中心医院的非必要复诊减少。不过，不同机构的电子病历格式不一致，影像资料传输也存在延迟。项目组统一了基础数据字段，并明确远程会诊责任和转诊标准。评估认为，联合门诊能改善连续照护，但前提是基层人员稳定、设备可用且患者隐私得到保护，不能把技术平台当作替代基层能力建设的捷径。",
    ),
    (
        "heritage_digitization",
        "文化遗产数字化",
        "一座地方博物馆对脆弱文物和传统工艺进行数字化记录。团队使用高分辨率摄影、三维扫描和口述史访谈建立档案，并计划向学校和研究者开放部分资源。数字展览扩大了访问范围，但也出现文件格式不统一、元数据缺失和授权边界模糊的问题。传承人担心工艺细节被商业复制，因此项目增加分级访问和用途许可。博物馆还制定多地备份和格式迁移计划，以避免设备更新后无法读取。总结指出，数字化不是简单拍照，长期价值取决于描述标准、权利协商、保存机制和持续维护。",
    ),
    (
        "grid_storage",
        "电网储能示范",
        "某省在风电集中地区建设电池储能示范站，用于平滑短时功率波动和参与调峰。运行数据显示，储能可以在风速突然变化时快速响应，也能在用电高峰释放电量。但项目收益受到电价机制、循环寿命和安全维护成本影响。一次局部过热事件后，运营方增加电芯级监测、分区隔离和消防演练，并调整充放电深度。专家认为，储能价值不能只按峰谷价差计算，还包括备用、频率调节和延缓电网扩建等服务。后续推广需要明确市场补偿和全生命周期回收责任。",
    ),
    (
        "small_business_finance",
        "中小企业融资平台",
        "某地上线中小企业融资服务平台，将税务、合同和应收账款信息用于辅助银行评估。部分缺少抵押物的企业因此获得了流动资金贷款，平均审批时间也有所缩短。不过，小微企业担心数据使用范围不透明，银行则指出单一平台数据可能无法反映经营突变。监管部门要求平台记录每次数据调用，并提供异议和更正渠道，同时禁止把与还款能力无关的信息用于评分。试点评估认为，数据可以降低信息不对称，但不能完全替代人工尽调和风险分担机制，还要防止算法把历史弱势固化为持续拒贷。",
    ),
    (
        "online_learning",
        "在线课程教学改进",
        "一所高校对大规模在线课程进行教学改进。原课程主要由长视频和期末考试组成，学生完成率较低。教师将内容拆分为短讲解、即时练习和每周项目，并增加同伴互评与答疑直播。新版本中，练习参与率提高，但同伴互评质量差异较大，部分学生也因时区和网络条件无法参加直播。团队随后提供评分示例、异步讨论和低带宽材料。研究发现，频繁互动有助于保持学习节奏，但过多通知会增加负担。课程设计需要在结构化支持、自主安排和真实任务之间取得平衡。",
    ),
    (
        "waste_sorting",
        "城市垃圾分类优化",
        "某城区在垃圾分类实施两年后重新评估运行方式。早期依靠大量现场督导，准确率提升较快，但督导减少后部分小区出现反弹。调查发现，投放点距离、标识一致性和物业清运方式比单次宣传更影响居民行为。城区随后统一容器颜色，调整投放时间，并公开分类后运输去向。对餐饮商户则增加油水分离和厨余称重服务。评估认为，分类效果不仅取决于居民，也取决于收集、运输和处理是否真正分流；如果末端混运，会迅速损害公众信任。",
    ),
    (
        "research_collaboration",
        "跨地区科研协作",
        "一个跨地区科研团队研究空气污染与健康关系，成员来自五所大学和两家医院。项目初期，各单位的数据格式、伦理审批和分析习惯不同，导致合并进度缓慢。团队建立统一数据字典、版本控制和预注册分析方案，并规定敏感数据只能在受控环境中访问。协作效率提高后，又出现年轻研究者贡献难以被论文署名充分体现的问题。项目因此记录数据整理、软件开发和项目管理贡献，并在研究开始前讨论署名原则。总结认为，远程工具可以降低沟通成本，但高质量合作仍依赖清晰责任、可信治理和对隐性劳动的认可。",
    ),
    (
        "smart_agriculture",
        "智慧农业灌溉试点",
        "一个缺水地区在果园开展智慧灌溉试点。土壤湿度传感器、气象预报和作物生长阶段共同决定每日灌水量，农户可以通过手机查看建议。试点果园用水量下降，产量总体稳定，但部分传感器因盐碱和维护不足出现漂移，造成错误建议。技术团队增加人工校准和异常报警，并允许农户覆盖自动决策。调查发现，大户更容易承担设备成本，小户则偏好合作社共享服务。项目建议将节水效果、设备可靠性和农户经验结合评估，避免把模型输出当作不可质疑的命令。",
    ),
    (
        "tourism_capacity",
        "景区旅游承载管理",
        "某山地景区在节假日面临道路拥堵、步道侵蚀和垃圾激增。管理方引入分时预约，并根据停车位、天气和步道监测动态调整入园额度。高峰期排队时间下降，但周边商户担心客流减少，游客也抱怨临时额度变化。景区随后提前公布基础额度，只在极端天气下调整，并引导游客前往较少使用的路线。生态监测显示，部分核心区域压力减轻，但替代路线需要加强维护。报告认为，承载量不是一个永久固定数字，应同时考虑生态恢复速度、基础设施、安全能力和社区收益。",
    ),
]

CODE_FAMILIES = [
    ("lru_cache", "实现一个容量固定、支持 get 和 put 的 LRU Cache。"),
    ("retry_decorator", "实现一个支持最大重试次数和指数退避的 Python 重试装饰器。"),
    ("log_parser", "实现一个解析结构化日志并按错误级别统计数量的工具。"),
    ("rate_limiter", "实现一个线程安全的令牌桶限流器。"),
    ("csv_aggregator", "实现一个读取 CSV 并按指定列分组求和的工具。"),
    ("dependency_resolver", "实现一个根据依赖关系返回合法构建顺序的函数。"),
    ("ttl_cache", "实现一个支持过期时间和惰性清理的 TTL Cache。"),
    ("task_scheduler", "实现一个按优先级和提交时间调度任务的队列。"),
    ("config_validator", "实现一个校验嵌套配置字段、类型和必填项的工具。"),
    ("file_deduplicator", "实现一个根据文件内容哈希查找重复文件的工具。"),
    ("event_bus", "实现一个支持订阅、退订和事件发布的进程内 Event Bus。"),
    ("circuit_breaker", "实现一个具有 closed、open、half-open 状态的熔断器。"),
    ("text_diff", "实现一个按行比较两个文本并输出新增、删除和未变行的工具。"),
    ("pagination_iterator", "实现一个自动请求后续页面的惰性分页迭代器。"),
    ("metrics_window", "实现一个维护滑动时间窗口并计算平均值和分位数的组件。"),
    ("inventory_allocator", "实现一个按订单优先级分配有限库存的函数。"),
    ("permission_checker", "实现一个支持角色继承和显式拒绝规则的权限检查器。"),
    ("markdown_toc", "实现一个从 Markdown 标题生成层级目录的工具。"),
    ("async_worker_pool", "实现一个限制并发数并收集任务结果的异步工作池。"),
    ("expression_evaluator", "实现一个安全计算四则运算和括号表达式的解析器。"),
]


def _qa_prompts(question: str) -> dict[str, str]:
    return {
        "short": f"请用一到两句话直接回答下面的问题，不要列举扩展内容：\n{question}",
        "medium": f"请用三到五段回答下面的问题，解释核心机制，并给出一个具体例子：\n{question}",
        "long": (
            "请围绕下面的问题写一篇结构化教程。内容至少包括概念定义、工作机制、"
            f"具体例子、常见误区、应用边界和总结：\n{question}"
        ),
    }


def _summary_prompts(title: str, source: str) -> dict[str, str]:
    material = f"材料标题：{title}\n\n{source}"
    return {
        "short": f"请把以下材料压缩成一句话摘要，只保留最核心结论。\n\n{material}",
        "medium": f"请把以下材料总结为五到八条要点，覆盖背景、措施和主要结果。\n\n{material}",
        "long": (
            "请对以下材料做详细的结构化总结，按背景、问题、实施措施、证据与结果、"
            f"风险与局限、最终结论分节组织，并保留重要数字和因果关系。\n\n{material}"
        ),
    }


def _code_prompts(requirement: str) -> dict[str, str]:
    return {
        "short": (
            f"请使用 Python 完成下面的任务。只输出核心实现，不要解释、示例或测试。\n{requirement}"
        ),
        "medium": (
            "请使用 Python 完成下面的任务，包含类型标注、docstring、输入校验和主要边界处理，"
            f"并在代码后简要说明实现思路。\n{requirement}"
        ),
        "long": (
            "请把下面的任务实现为完整、可运行的 Python 模块。回答必须包含核心实现、异常处理、"
            "单元测试、使用示例、时间与空间复杂度分析，以及关键设计选择说明。\n"
            f"{requirement}"
        ),
    }


def _family_definitions() -> dict[str, list[tuple[str, dict[str, str]]]]:
    return {
        "qa": [(slug, _qa_prompts(question)) for slug, question in QA_FAMILIES],
        "summarization": [
            (slug, _summary_prompts(title, source)) for slug, title, source in SUMMARY_FAMILIES
        ],
        "code": [(slug, _code_prompts(requirement)) for slug, requirement in CODE_FAMILIES],
    }


def build_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    rng = random.Random(SPLIT_SEED)
    for task, families in _family_definitions().items():
        family_ids = [f"{task}_{index:03d}_{slug}" for index, (slug, _) in enumerate(families, 1)]
        shuffled = family_ids.copy()
        rng.shuffle(shuffled)
        test_families = set(shuffled[:4])
        for family_id, (_, prompts) in zip(family_ids, families, strict=True):
            split = "test" if family_id in test_families else "train"
            for length_class in LENGTH_ORDER:
                records.append(
                    {
                        "dataset_version": DATASET_VERSION,
                        "prompt_family_id": family_id,
                        "prompt_id": f"{family_id}_{length_class}",
                        "task_type": task,
                        "intended_length": length_class,
                        "intended_output_tokens": LENGTH_RANGES[length_class],
                        "language": "zh-CN",
                        "split": split,
                        "generation_seeds": [42, 43, 44],
                        "prompt": prompts[length_class],
                    }
                )
    validate_records(records)
    return records


def validate_records(records: list[dict[str, object]]) -> None:
    if len(records) != 180:
        raise ValueError(f"expected 180 prompts, found {len(records)}")
    if len({record["prompt_id"] for record in records}) != len(records):
        raise ValueError("prompt_id values must be unique")

    task_counts = Counter(record["task_type"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    if task_counts != {"qa": 60, "summarization": 60, "code": 60}:
        raise ValueError(f"unexpected task counts: {task_counts}")
    if split_counts != {"train": 144, "test": 36}:
        raise ValueError(f"unexpected split counts: {split_counts}")

    families: dict[object, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        families[record["prompt_family_id"]].append(record)
    if len(families) != 60:
        raise ValueError(f"expected 60 families, found {len(families)}")
    for family_id, family_records in families.items():
        if len(family_records) != 3:
            raise ValueError(f"family {family_id} must contain three prompts")
        if {record["intended_length"] for record in family_records} != set(LENGTH_ORDER):
            raise ValueError(f"family {family_id} is missing a length variant")
        if len({record["split"] for record in family_records}) != 1:
            raise ValueError(f"family {family_id} crosses splits")


def write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts/alps_v1_prompts.jsonl"),
    )
    args = parser.parse_args()
    records = build_records()
    output = write_manifest(args.output, records)
    print(f"wrote {len(records)} prompts to {output}")


if __name__ == "__main__":
    main()
