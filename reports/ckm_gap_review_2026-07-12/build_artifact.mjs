import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const markdown = fs.readFileSync(path.join(here, "report.md"), "utf8");
const title = "CKM 相关论文进展与研究 GAP 调研";
const body = markdown.replace(/^# CKM 相关论文进展与研究 GAP 调研\s*/u, "");
const issueTableStart = "| 序号 | 原文件问题 | 截至 2026-07 的论文进展 | GAP 判定 | 建议如何改写 |";
const issueTableEnd = "\n### 成熟度摘要";
const planTableStart = "| 模块 | 核心科学问题 | 对应原文件 | 建议产出 |";
const planTableEnd = "\n\n优先级建议：";
let artifactBody = body;
const issueStartIndex = artifactBody.indexOf(issueTableStart);
const issueEndIndex = artifactBody.indexOf(issueTableEnd, issueStartIndex);
if (issueStartIndex >= 0 && issueEndIndex > issueStartIndex) {
  artifactBody = `${artifactBody.slice(0, issueStartIndex)}<!-- NATIVE_ISSUE_TABLE -->${artifactBody.slice(issueEndIndex)}`;
}
const planStartIndex = artifactBody.indexOf(planTableStart);
const planEndIndex = artifactBody.indexOf(planTableEnd, planStartIndex);
if (planStartIndex >= 0 && planEndIndex > planStartIndex) {
  artifactBody = `${artifactBody.slice(0, planStartIndex)}<!-- NATIVE_PLAN_TABLE -->${artifactBody.slice(planEndIndex)}`;
}
const sectionMarker = "\n## 11 项问题的逐项判定";
const markerIndex = artifactBody.indexOf(sectionMarker);
const introBody = markerIndex >= 0 ? artifactBody.slice(0, markerIndex) : artifactBody;
const detailBody = markerIndex >= 0 ? artifactBody.slice(markerIndex) : "";
const issueToken = "<!-- NATIVE_ISSUE_TABLE -->";
const planToken = "<!-- NATIVE_PLAN_TABLE -->";
const [beforeIssueTable, afterIssueToken = ""] = detailBody.split(issueToken);
const [betweenTables, afterPlanTable = ""] = afterIssueToken.split(planToken);
const generatedAt = new Date().toISOString();

const sources = [
  {
    id: "source_document",
    label: "《课题四问题梳理与解决计划》11项问题提取",
    path: "source_document_extract.md",
    query: {
      engine: "local-file",
      description: "从用户提供的 DOCX OOXML 正文提取并人工归并为 11 个问题",
      executed_at: generatedAt
    }
  },
  {
    id: "source_notes",
    label: "检索方法、证据边界与代表性来源清单",
    path: "source_notes.md",
    query: {
      engine: "source-review",
      description: "围绕11个问题的目标化一手文献与官方标准目录核验；截止2026-07-12",
      executed_at: generatedAt
    }
  },
  {
    id: "issue_maturity_sql",
    label: "11项问题成熟度分类派生数据",
    path: "issue_maturity.sql",
    query: {
      engine: "sql-values",
      sql: "SELECT * FROM (VALUES ('强GAP', 5, '1、3、5、10、11', '仍有实质空白，但需以可验证边界重写'), ('需收束', 3, '4、6、7', '相关模块已推进，应转向交叉或闭环问题'), ('原表述已占据', 3, '2、8、9', '宽泛创新点已有高度重合工作')) AS issue_maturity(category, count, items, interpretation);",
      description: "由11项逐条证据审查结果汇总的透明分类数据；非论文计量",
      executed_at: generatedAt
    }
  },
  {
    id: "issue_review_sql",
    label: "11项问题逐项证据判定",
    path: "issue_review.sql"
  },
  {
    id: "paper_plan_sql",
    label: "建议的课题与论文结构",
    path: "paper_plan.sql"
  },
  {
    id: "foundational_ckm",
    label: "Toward Environment-Aware 6G Communications via CKM",
    href: "https://arxiv.org/abs/2007.09332",
    query: { engine: "paper", description: "CKM定义与早期应用边界", executed_at: generatedAt }
  },
  {
    id: "ckm_tutorial",
    label: "A Tutorial on Environment-Aware Communications via CKM for 6G",
    href: "https://arxiv.org/abs/2309.07460",
    query: { engine: "paper", description: "CKM体系、构建、查询、应用与开放问题", executed_at: generatedAt }
  },
  {
    id: "ckmdiff",
    label: "CKMDiff, npj Wireless Technology, 2026",
    href: "https://www.nature.com/articles/s44459-026-00042-1",
    query: { engine: "journal", description: "生成式CKM去噪、补全和超分辨率的近期同行评审证据", executed_at: generatedAt }
  },
  {
    id: "pmnet",
    label: "A Scalable and Generalizable Pathloss Map Prediction",
    href: "https://arxiv.org/abs/2312.03950",
    query: { engine: "paper", description: "跨城市迁移与目标域微调边界", executed_at: generatedAt }
  },
  {
    id: "indoor_challenge",
    label: "The First Indoor Pathloss Radio Map Prediction Challenge",
    href: "https://arxiv.org/abs/2501.13698",
    query: { engine: "paper", description: "跨室内布局、频率和天线方向图benchmark", executed_at: generatedAt }
  },
  {
    id: "ckmimagenet",
    label: "CKMImageNet",
    href: "https://arxiv.org/abs/2504.09849",
    query: { engine: "paper", description: "多维CKM数据对象与数据集边界", executed_at: generatedAt }
  },
  {
    id: "spectrum_surveying",
    label: "Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs",
    href: "https://arxiv.org/abs/2201.04125",
    query: { engine: "paper", description: "不确定度驱动主动测量与轨迹的先行工作", executed_at: generatedAt }
  },
  {
    id: "bayes_active",
    label: "Bayesian Active Learning for Sample Efficient 5G Radio Map Reconstruction",
    href: "https://doi.org/10.1109/TWC.2024.3483112",
    query: { engine: "paper", description: "真实5G数据、主动选点与移动代价", executed_at: generatedAt }
  },
  {
    id: "learning_radio_environments",
    label: "Learning Radio Environments by Differentiable Ray Tracing",
    href: "https://arxiv.org/abs/2311.18558",
    query: { engine: "paper", description: "真实测量下的材料、散射和天线参数学习", executed_at: generatedAt }
  },
  {
    id: "mef_update",
    label: "Update Strategy for Channel Knowledge Map in Complex Environments",
    href: "https://arxiv.org/abs/2512.15154",
    query: { engine: "paper", description: "Map Efficacy Function与CKM更新调度", executed_at: generatedAt }
  },
  {
    id: "dynamic_map",
    label: "A Novel 6G Dynamic Channel Map Based on a Hybrid Channel Model",
    href: "https://arxiv.org/abs/2604.15083",
    query: { engine: "paper", description: "真实测量支持的RT-GBSM快速动态地图", executed_at: generatedAt }
  },
  {
    id: "angle_domain_ma",
    label: "Channel Map-Based Angle Domain Multiple Access",
    href: "https://shuaifeichen273.github.io/paper/2025_JSTSP_Channel_Map-Based_Angle_Domain_Multiple_Access_for_Cell-Free_Massive_MIMO_Communications.pdf",
    query: { engine: "paper", description: "CKM谱系下的多用户接收、导频、LSFD与AP选择", executed_at: generatedAt }
  },
  {
    id: "biwgs",
    label: "6D CKM Construction via Bidirectional Wireless Gaussian Splatting",
    href: "https://arxiv.org/abs/2510.26166",
    query: { engine: "paper", description: "连续三维Tx-Rx位置查询及其物理边界", executed_at: generatedAt }
  },
  {
    id: "capa_tutorial",
    label: "Electromagnetic Signal and Information Theory: A Continuous-Aperture Array Perspective",
    href: "https://arxiv.org/abs/2605.12910",
    query: { engine: "paper", description: "连续孔径传播算子、容量与自由度边界", executed_at: generatedAt }
  },
  {
    id: "3gpp_aiml",
    label: "3GPP TR 38.843: Study on AI/ML for NR Air Interface",
    href: "https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3983",
    query: { engine: "official-standard", description: "核验CKM相邻AI/ML标准而非CKM专用语义", executed_at: generatedAt }
  },
  {
    id: "itu_imt2030",
    label: "ITU-R M.2160: IMT-2030 Framework",
    href: "https://www.itu.int/rec/R-REC-M.2160-0-202311-I",
    query: { engine: "official-standard", description: "核验IMT-2030总体框架与CKM接口边界", executed_at: generatedAt }
  }
];

const artifact = {
  surface: "report",
  manifest: {
    version: 1,
    surface: "report",
    title,
    description: "基于用户课题文件的11项问题，对截至2026-07-12的CKM论文、原型、跨域验证和标准化进展进行逐项GAP审查。",
    generatedAt,
    cards: [],
    charts: [
      {
        id: "issue_maturity_chart",
        title: "11项问题的研究成熟度分类",
        subtitle: "按原问题表述的新颖性风险分类；这是证据审查结果，不是论文计量。",
        headerMarkdown: "强GAP仍需明确问题边界；“需收束”表示相关模块已有明显进展；“原表述已占据”表示不能再按原措辞宣称首创。",
        type: "bar",
        dataset: "issue_maturity",
        sourceId: "issue_maturity_sql",
        valueFormat: "number",
        encodings: {
          x: { field: "category", type: "nominal", label: "判定类别" },
          y: { field: "count", type: "quantitative", label: "问题数量" },
          tooltip: [
            { field: "items", type: "nominal", label: "对应问题" },
            { field: "interpretation", type: "nominal", label: "含义" }
          ]
        }
      }
    ],
    tables: [
      {
        id: "issue_review_table",
        title: "11项问题逐项判定",
        subtitle: "每一项均区分已有进展、剩余GAP与建议改写；可按序号排序。",
        headerMarkdown: "“已出现相关论文”不等于问题完全解决；判定关注的是原措辞还能否作为新颖性主张。",
        dataset: "issue_review",
        sourceId: "issue_review_sql",
        defaultSort: { field: "item_no", direction: "asc" },
        density: "dense",
        columns: [
          { field: "item_no", label: "序号", format: "number" },
          { field: "question", label: "原文件问题", type: "text" },
          { field: "progress", label: "论文进展", type: "text" },
          { field: "gap_status", label: "GAP判定", type: "text" },
          { field: "rewrite", label: "建议改写", type: "text" }
        ]
      },
      {
        id: "paper_plan_table",
        title: "建议的课题与论文结构",
        subtitle: "四个模块共享同一数据协议与评测闭环。",
        dataset: "paper_plan",
        sourceId: "paper_plan_sql",
        defaultSort: { field: "module", direction: "asc" },
        density: "spacious",
        columns: [
          { field: "module", label: "模块", type: "text" },
          { field: "scientific_question", label: "核心科学问题", type: "text" },
          { field: "source_items", label: "对应原文件", type: "text" },
          { field: "deliverable", label: "建议产出", type: "text" }
        ]
      }
    ],
    sources,
    blocks: [
      { id: "title", type: "markdown", body: `# ${title}` },
      { id: "executive_summary", type: "markdown", body: introBody },
      { id: "issue_maturity", type: "chart", chartId: "issue_maturity_chart" },
      { id: "issue_review_intro", type: "markdown", body: beforeIssueTable },
      { id: "issue_review", type: "table", tableId: "issue_review_table" },
      { id: "main_analysis", type: "markdown", body: betweenTables },
      { id: "paper_plan", type: "table", tableId: "paper_plan_table" },
      { id: "closing", type: "markdown", body: afterPlanTable }
    ]
  },
  snapshot: {
    version: 1,
    generatedAt,
    status: "ready",
    datasets: {
      issue_maturity: [
        {
          category: "强GAP",
          count: 5,
          items: "1、3、5、10、11",
          interpretation: "仍有实质空白，但需以可验证边界重写"
        },
        {
          category: "需收束",
          count: 3,
          items: "4、6、7",
          interpretation: "相关模块已推进，应转向交叉或闭环问题"
        },
        {
          category: "原表述已占据",
          count: 3,
          items: "2、8、9",
          interpretation: "宽泛创新点已有高度重合工作"
        }
      ],
      issue_review: [
        { item_no: 1, question: "城市、郊区、室内分别建模后形成统一 CKM", progress: "跨城市、跨布局和少样本迁移已有进展，多数仍局限二维 path gain、同一仿真体系或目标域微调", gap_status: "强 GAP，但必须具体化", rewrite: "多维连续 CKM + 环境不变表示/场景低维适配 + 室内—城市—乡村及仿真到实测联合验证" },
        { item_no: 2, question: "在预测误差大或聚类中心处选点并规划轨迹", progress: "不确定度驱动 UAV 路径、真实 5G 主动采样、生成模型选点和 utility-aware path 已出现", gap_status: "原核心思路已被占据", rewrite: "联合地图误差、变化检测、参数可辨识度、任务效用、能耗和安全约束" },
        { item_no: 3, question: "用信道数据校准环境几何和材料电磁参数", progress: "材料、散射和天线参数可微校准已有真实测量；复杂网格几何通常固定", gap_status: "材料较成熟；联合在线反演仍有强 GAP", rewrite: "局部几何—材料联合反演、误差归因、跨频一致性、可辨识性和置信区间" },
        { item_no: 4, question: "用实时数据更新几何和材料、维护 CKM", progress: "动态路径分离、RT+GBSM、时序预测、局部刷新和 ISAC 补偿已有工作", gap_status: "部分解决", rewrite: "变化定位—局部解冻—联合更新几何/材料—长期真实闭环" },
        { item_no: 5, question: "CKM 降低导频/检测/预编码复杂度并支持同频 100 流", progress: "波束、导频、估计、角域多用户接收和调度已有工作；直接论文常见 1–40 流/用户", gap_status: "导频/检测已有进展；100 流实证仍是 GAP", rewrite: "定义流数口径并评估净 SE、P95 延迟、能耗、地图错误和回退训练" },
        { item_no: 6, question: "连续立体空间模型及空间/时间/频率外的新自由度", progress: "6D X2X CKM、RF 场、连续孔径/全息 MIMO 和近场 DoF 已分别发展", gap_status: "交叉方向有 GAP；创造新自由度表述不成立", rewrite: "学习和调度近场距离聚焦、极化及连续孔径电磁模态" },
        { item_no: 7, question: "比较新旧模型下的传输性能", progress: "RMSE/NMSE、SE、EE、BLER 等指标分散，常忽略建图和更新成本", gap_status: "非独立创新；统一评测仍有 GAP", rewrite: "信道误差—可信度—任务收益—生命周期成本四层 benchmark" },
        { item_no: 8, question: "CKM 查询/推理开销；做压缩、低秩或稀疏化", progress: "低秩、稀疏采样、生成加速和 Gaussian Splatting 剪枝已有大量工作", gap_status: "宽泛压缩 CKM 已被占据", rewrite: "任务最小充分 CKM 与更新—存储—查询—空口净收益联合优化" },
        { item_no: 9, question: "设计类似 AoI 的及时性/可靠性指标和更新策略", progress: "Map Efficacy Function 已批评经典 AoI 并优化 CKM 更新时间", gap_status: "原表述基本被占据", rewrite: "在线学习的任务级/区域级可靠度，联合更新时机、区域、采样和轨迹" },
        { item_no: 10, question: "不同设备/系统间的 CKM 通用表示与转换", progress: "已有 REKP、RF radiance field、数据集和 foundation representation；无专用规范语义层", gap_status: "强 GAP", rewrite: "定义物理多径对象、查询键、坐标/单位、置信度、有效期、来源、版本和能力协商" },
        { item_no: 11, question: "标准化评估 CKM 的资源、响应、能耗和精度", progress: "有挑战赛和论文级指标，但数据、切分、硬件及系统假设碎片化", gap_status: "强 GAP，但列指标不够", rewrite: "公开协议、跨域切分、生命周期核算、硬件测试规程和任务门限" }
      ],
      paper_plan: [
        { module: "P1 统一表示", scientific_question: "哪些传播知识跨环境不变，哪些只需少量局部参数适配？", source_items: "1、10", deliverable: "多维物理对象/低秩模态表示；跨域 benchmark" },
        { module: "P2 主动校准", scientific_question: "哪组测量最大化变化检测与几何—材料可辨识度，同时满足轨迹约束？", source_items: "2、3", deliverable: "联合 acquisition + trajectory；局部反演与不确定度" },
        { module: "P3 动态维护", scientific_question: "何时、哪里、更新什么，才能最大化任务效用并控制生命周期成本？", source_items: "4、8、9", deliverable: "空间化 MEF/可靠度；事件触发增量更新" },
        { module: "P4 通信闭环", scientific_question: "CKM 在导频—检测—预编码中的净收益能否随流数稳定扩展？", source_items: "5、6、7、11", deliverable: "4→16→40→100+ 流标度实验；硬件/测量验证" }
      ]
    }
  },
  sources
};

fs.writeFileSync(path.join(here, "artifact.json"), `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
