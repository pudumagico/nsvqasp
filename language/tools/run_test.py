import os
import json

from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils
import torch

def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    # print(out_mod)
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    elif out_mod.startswith('between_projection'):
        q_type = 'between_projection'
    elif out_mod.startswith('between_bbox'):
        q_type = 'between_bbox'
    elif out_mod.startswith('between_proper'):
        q_type = 'between_proper'
    else:
        q_type = out_mod
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True

torch.backends.cudnn.benchmark = True

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
model = Seq2seqParser(opt)
print('| running test')
stats = {
    'count': 0,
    'count_tot': 0,
    'exist': 0,
    'exist_tot': 0,
    'compare_num': 0,
    'compare_num_tot': 0,
    'compare_attr': 0,
    'compare_attr_tot': 0,
    'query': 0,
    'query_tot': 0,
    'correct_ans': 0,
    'correct_prog': 0,
    'total': 0,
    'between_projection_tot': 0,
    'between_bbox_tot': 0,
    'between_proper_tot': 0,
    'not_between_projection_tot': 0,
    'not_between_bbox_tot': 0,
    'boolean_negation_tot': 0,
    'boolean_negation': 0,
    'between_projection': 0,
    'between_bbox': 0,
    'between_proper': 0,
    'between_projection_tot': 0,
    'between_bbox_tot': 0,
    'all_different': 0,
    'all_different_tot': 0,
    'two_equal': 0,
    'two_equal_tot': 0,
    'three_equal': 0,
    'three_equal_tot': 0,
    'four_equal': 0,
    'four_equal_tot': 0,
    'exactly_two_equal': 0,
    'exactly_two_equal_tot': 0,
    'exactly_three_equal': 0,
    'exactly_three_equal_tot': 0,
    'exactly_four_equal': 0,
    'exactly_four_equal_tot': 0,
    'count_between_projection': 0,
    'count_between_projection_tot': 0,
    'count_not_between_projection': 0,
    'count_not_between_projection_tot': 0,
    'count_between_bbox': 0,
    'count_between_bbox_tot': 0,
    'count_not_between_bbox': 0,
    'count_not_between_bbox_tot': 0,
    'count_between_proper': 0,
    'count_between_proper_tot': 0,
    'count_not_between_proper': 0,
    'count_not_between_proper_tot': 0,
    'count_different': 0,
    'count_different_tot': 0,
    'set_difference': 0,
    'set_difference_tot': 0

}
for x, y, ans, idx in loader:
    model.set_input(x, y)
    pred_program = model.parse()
    y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()

    for i in range(pg_np.shape[0]):
        pred_ans = executor.run(pg_np[i], idx_np[i], 'val', guess=True)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

        q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][y_np[i][1]])
        if q_type == 'set_difference':
            print(pred_ans, gt_ans)
            
        if pred_ans == gt_ans:
            stats[q_type] += 1
            stats['correct_ans'] += 1
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1
        # print(q_type, gt_ans, pred_ans)
        stats['%s_tot' % q_type] += 1
        stats['total'] += 1
    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

    # print('between_projection', stats['between_projection']/ stats['between_projection_tot'])
    # print('between_bbox', stats['between_bbox']/ stats['between_bbox_tot'])
    # print('between_proper', stats['between_proper']/ stats['between_proper_tot'])
    # print('boolean_negation', stats['boolean_negation']/ stats['boolean_negation_tot'])

    # print('two_equal', stats['exactly_two_equal']/ stats['exactly_two_equal_tot'])
    # print('three_equal', stats['three_equal']/ stats['three_equal_tot'])
    # print('four_equal', stats['four_equal']/ stats['four_equal_tot'])
    # print('exactly_two_equal', stats['exactly_two_equal']/ stats['exactly_two_equal_tot'])
    # print('exactly_three_equal', stats['exactly_three_equal']/ stats['exactly_three_equal_tot'])
    # print('exactly_four_equal', stats['exactly_four_equal']/ stats['exactly_four_equal_tot'])
    # print('all_different', stats['all_different']/ stats['all_different_tot'])

    print('count_between_projection', stats['count_between_projection']/ stats['count_between_projection_tot'])
    print('count_between_bbox', stats['count_between_bbox']/ stats['count_between_bbox_tot'])
    print('count_between_proper', stats['count_between_proper']/ stats['count_between_proper_tot'])
    print('set_difference', stats['set_difference']/ stats['set_difference_tot'])
    print('count_different', stats['count_different']/ stats['count_different_tot'])


result = {
    'count_acc': stats['count'] / stats['count_tot'],
    'exist_acc': stats['exist'] / stats['exist_tot'],
    'compare_num_acc': stats['compare_num'] / stats ['compare_num_tot'],
    'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
    'query_acc': stats['query'] / stats['query_tot'],
    'program_acc': stats['correct_prog'] / stats['total'],
    'overall_acc': stats['correct_ans'] / stats['total']
}
print(result)

utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    