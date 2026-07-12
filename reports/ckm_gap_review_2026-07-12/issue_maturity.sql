-- Evidence classification of the 11 source-document questions.
-- This is a transparent derived dataset, not a literature-count query.
SELECT *
FROM (VALUES
  ('强GAP', 5, '1、3、5、10、11', '仍有实质空白，但需以可验证边界重写'),
  ('需收束', 3, '4、6、7', '相关模块已推进，应转向交叉或闭环问题'),
  ('原表述已占据', 3, '2、8、9', '宽泛创新点已有高度重合工作')
) AS issue_maturity(category, count, items, interpretation);
