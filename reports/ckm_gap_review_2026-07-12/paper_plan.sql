-- Recommended paper structure synthesized from the source-document questions.
SELECT *
FROM (VALUES
  ('P1 统一表示', '哪些传播知识跨环境不变，哪些只需少量局部参数适配？', '1、10', '多维物理对象/低秩模态表示；跨域 benchmark'),
  ('P2 主动校准', '哪组测量最大化变化检测与几何—材料可辨识度，同时满足轨迹约束？', '2、3', '联合 acquisition + trajectory；局部反演与不确定度'),
  ('P3 动态维护', '何时、哪里、更新什么，才能最大化任务效用并控制生命周期成本？', '4、8、9', '空间化 MEF/可靠度；事件触发增量更新'),
  ('P4 通信闭环', 'CKM 在导频—检测—预编码中的净收益能否随流数稳定扩展？', '5、6、7、11', '4→16→40→100+ 流标度实验；硬件/测量验证')
) AS paper_plan(module, scientific_question, source_items, deliverable);
