import math
import os
import json
import numpy as np
import torch
import re

import clingo

id2cat = {
  "0": "LaGraMeCy",
  "1": "LaGraMeSp",
  "2": "LaGraMeCu",
  "3": "LaGraRuCy",
  "4": "LaGraRuSp",
  "5": "LaGraRuCu",
  "6": "LaBlMeCy",
  "7": "LaBlMeSp",
  "8": "LaBlMeCu",
  "9": "LaBlRuCy",
  "10": "LaBlRuSp",
  "11": "LaBlRuCu",
  "12": "LaBrMeCy",
  "13": "LaBrMeSp",
  "14": "LaBrMeCu",
  "15": "LaBrRuCy",
  "16": "LaBrRuSp",
  "17": "LaBrRuCu",
  "18": "LaYeMeCy",
  "19": "LaYeMeSp",
  "20": "LaYeMeCu",
  "21": "LaYeRuCy",
  "22": "LaYeRuSp",
  "23": "LaYeRuCu",
  "24": "LaReMeCy",
  "25": "LaReMeSp",
  "26": "LaReMeCu",
  "27": "LaReRuCy",
  "28": "LaReRuSp",
  "29": "LaReRuCu",
  "30": "LaGreMeCy",
  "31": "LaGreMeSp",
  "32": "LaGreMeCu",
  "33": "LaGreRuCy",
  "34": "LaGreRuSp",
  "35": "LaGreRuCu",
  "36": "LaPuMeCy",
  "37": "LaPuMeSp",
  "38": "LaPuMeCu",
  "39": "LaPuRuCy",
  "40": "LaPuRuSp",
  "41": "LaPuRuCu",
  "42": "LaCyMeCy",
  "43": "LaCyMeSp",
  "44": "LaCyMeCu",
  "45": "LaCyRuCy",
  "46": "LaCyRuSp",
  "47": "LaCyRuCu",
  "48": "SmGraMeCy",
  "49": "SmGraMeSp",
  "50": "SmGraMeCu",
  "51": "SmGraRuCy",
  "52": "SmGraRuSp",
  "53": "SmGraRuCu",
  "54": "SmBlMeCy",
  "55": "SmBlMeSp",
  "56": "SmBlMeCu",
  "57": "SmBlRuCy",
  "58": "SmBlRuSp",
  "59": "SmBlRuCu",
  "60": "SmBrMeCy",
  "61": "SmBrMeSp",
  "62": "SmBrMeCu",
  "63": "SmBrRuCy",
  "64": "SmBrRuSp",
  "65": "SmBrRuCu",
  "66": "SmYeMeCy",
  "67": "SmYeMeSp",
  "68": "SmYeMeCu",
  "69": "SmYeRuCy",
  "70": "SmYeRuSp",
  "71": "SmYeRuCu",
  "72": "SmReMeCy",
  "73": "SmReMeSp",
  "74": "SmReMeCu",
  "75": "SmReRuCy",
  "76": "SmReRuSp",
  "77": "SmReRuCu",
  "78": "SmGreMeCy",
  "79": "SmGreMeSp",
  "80": "SmGreMeCu",
  "81": "SmGreRuCy",
  "82": "SmGreRuSp",
  "83": "SmGreRuCu",
  "84": "SmPuMeCy",
  "85": "SmPuMeSp",
  "86": "SmPuMeCu",
  "87": "SmPuRuCy",
  "88": "SmPuRuSp",
  "89": "SmPuRuCu",
  "90": "SmCyMeCy",
  "91": "SmCyMeSp",
  "92": "SmCyMeCu",
  "93": "SmCyRuCy",
  "94": "SmCyRuSp",
  "95": "SmCyRuCu"
}

short2long = {
  "sizes": {
    "La": "large",
    "Sm": "small"
  },
  "shapes": {
    "Cy": "cylinder",
    "Sp": "sphere",
    "Cu": "cube"
  },
  "materials": {
    "Me": "metal",
    "Ru": "rubber"
  },
  "colors": {
    "Gra": "gray",
    "Bl": "blue",
    "Br": "brown",
    "Ye": "yellow",
    "Re": "red",
    "Gre": "green",
    "Pu": "purple",
    "Cy": "cyan"
  }
}

func_type = {
    "unary": ["scene", "unique", "count", "exist", "relate", "boolean_negation",
              "query_size", "query_color", "query_material", "query_shape",
              "two_equal", "three_equal", "four_equal", "all_different",
              "exactly_two_equal", "exactly_three_equal", "exactly_four_equal",
              "count_different", 
              "set_difference",
              "filter_size", "filter_color", "filter_material", "filter_shape", "same_size", "same_color", "same_material", "same_shape"],
    "binary": ["count_between_projection", "count_between_bbox", "count_between_proper",
            "equal_integer", "less_than", "greater_than", "equal_size", "equal_color", "equal_shape",
               "equal_material", "union", "intersect"],
    "ternary": ["between_projection", "between_bbox", "between_proper"]
}

def gt_scene(scene):
    obj_template = 'obj({},{},{},{},{},{},{},{}).'
    facts = ''
    id = 0
    for obj in scene['objects']:
        size = obj['size']
        color = obj['color']
        material = obj['material']
        shape = obj['shape']
        x1=obj['pixel_coords'][0]
        y1=obj['pixel_coords'][1]
        obj_fact = obj_template.format(0,id,size,color,material,shape,x1,y1)
        id += 1
        facts+=obj_fact
    return facts


def vision2facts(objects):
    # obj_template = 'obj({},{},{},{},{},{},{},{},{},{}).'
    obj_template = 'obj({},{},{},{},{},{},{},{}).'
    confidence_template = 'conf({},{}).'
    facts = ''
    id = 0
    scale = 1
    
    for obj in objects:
        # short_cat = id2cat[str(int(obj[0].item()))]
        # if obj[4] > 0.9:
        short_cat = obj[6]
        attrs = re.findall('[A-Z][^A-Z]*', short_cat)
        size = short2long['sizes'][attrs[0]]
        color = short2long['colors'][attrs[1]]
        material = short2long['materials'][attrs[2]]
        shape = short2long['shapes'][attrs[3]]

        # x1 = round((obj[0])*100)
        # y1 = round((obj[1])*100)
        # x2 = round((obj[2])*100)
        # y2 = round((obj[3])*100)

        # x1=round((x_center-width/2)*1000)
        # y1=round((y_center-heigth/2)*1000)
        # x2=round((x_center+width/2)*1000)
        # y2=round((y_center+heigth/2)*1000)

        x_center = round((obj[0]+obj[2])/2)
        y_center = round((obj[1]+obj[3])/2)
        # width = obj[3]
        # heigth = obj[4]
        # obj_fact = obj_template.format(0,id,size,color,material,shape,x1,y1,x2,y2)
        obj_fact = obj_template.format(0,id,size,color,material,shape,x_center,y_center)
        conf_fact = confidence_template.format(id,round(obj[4]*100))
        id += 1
        facts+=obj_fact+conf_fact
    return facts


def translate_tgt_tokens(encoding, vocab):
    predicate_q = ''
    for token in encoding:
        predicate_q += vocab['program_idx_to_token'][token.item()]
        predicate_q += ' '
    print('predicate form', predicate_q)
    return predicate_q


def translate_src_tokens(encoding, vocab):
    natural_q = ''
    for token in encoding:
        natural_q += vocab['question_idx_to_token'][token.item()]
        natural_q += ' '
    print('natural question', natural_q)
    return natural_q


def nodes_from_tree(tree, nodes=[]):
    if isinstance(tree, list):
        for child in tree:
            nodes_from_tree(child, nodes=nodes)
    else:
        nodes.append(tree)
    return nodes


def clean_facts(tree):
    if 'filter_size[small]' in tree:
        tree = tree.replace('filter_size[small]', 'filter_small')
    if 'filter_size[large]' in tree:
        tree = tree.replace('filter_size[large]', 'filter_large')
    if 'filter_color[gray]' in tree:
        tree = tree.replace('filter_color[gray]', 'filter_gray')
    if 'filter_color[blue]' in tree:
        tree = tree.replace('filter_color[blue]', 'filter_blue')
    if 'filter_color[red]' in tree:
        tree = tree.replace('filter_color[red]', 'filter_red')
    if 'filter_color[green]' in tree:
        tree = tree.replace('filter_color[green]', 'filter_green')
    if 'filter_color[brown]' in tree:
        tree = tree.replace('filter_color[brown]', 'filter_brown')
    if 'filter_color[purple]' in tree:
        tree = tree.replace('filter_color[purple]', 'filter_purple')
    if 'filter_color[cyan]' in tree:
        tree = tree.replace('filter_color[cyan]', 'filter_cyan')
    if 'filter_color[yellow]' in tree:
        tree = tree.replace('filter_color[yellow]', 'filter_yellow')
    if 'filter_material[metal]' in tree:
        tree = tree.replace('filter_material[metal]', 'filter_metal')
    if 'filter_material[rubber]' in tree:
        tree = tree.replace('filter_material[rubber]', 'filter_rubber')
    if 'filter_shape[cylinder]' in tree:
        tree = tree.replace('filter_shape[cylinder]', 'filter_cylinder')
    if 'filter_shape[cube]' in tree:
        tree = tree.replace('filter_shape[cube]', 'filter_cube')
    if 'filter_shape[sphere]' in tree:
        tree = tree.replace('filter_shape[sphere]', 'filter_sphere')
    if 'relate[behind]' in tree:
        tree = tree.replace('relate[behind]', 'relate_behind')
    if 'relate[front]' in tree:
        tree = tree.replace('relate[front]', 'relate_front')
    if 'relate[left]' in tree:
        tree = tree.replace('relate[left]', 'relate_left')
    if 'relate[right]' in tree:
        tree = tree.replace('relate[right]', 'relate_right')
    if 'filter_shape[sphere]' in tree:
        tree = tree.replace('filter_shape[sphere]', 'filter_sphere')
    if 'union' in tree:
        tree = tree.replace('union', 'or')
    if 'intersect' in tree:
        tree = tree.replace('intersect', 'and')

    return tree

def language2facts(encoding, vocab):
    encoding = [i for i in encoding if i != 0]
    try:
        tree = regenerate_tree([], encoding[:len(encoding)-1], len(encoding) - 1, vocab)
        facts = nodes_from_tree(tree, [])
        head_fact = facts[0]
        last_step = int(re.findall(r'\d+', head_fact)[0])
        return clean_facts(''.join(facts)) + f'end({last_step+1}).'
    except:
        return 'ans(translation_error).'


def regenerate_tree(tree, encoding, current_idx, vocab):
    
    token = vocab['program_idx_to_token'][encoding[0].item()]
    
    if any(p in token for p in func_type['unary']) and 'count_between' not in token:
        tree.append(token + '(' + str(current_idx-1) + ').')

        if encoding[1:]:
            only_branch = regenerate_tree([], encoding[1:], current_idx-1, vocab)
            tree.append(only_branch)

    elif any(q in token for q in func_type['binary']):

        scene_index = vocab['program_token_to_idx']['scene']
        scene_index = encoding.index(scene_index)

        tree.append(token + '(' + str(current_idx - 1) +
                    ',' + str(int(current_idx) - len(encoding[1:scene_index+1]) - 1) + ').')
        left_branch = regenerate_tree(
            [], encoding[1:scene_index+1], current_idx - 1, vocab)
        right_branch = regenerate_tree(
            [], encoding[scene_index+1:], current_idx - len(encoding[1:scene_index+1]) - 1, vocab)
        tree.append(left_branch)
        tree.append(right_branch)

    elif any(q in token for q in func_type['ternary']):
        
        scene_index = vocab['program_token_to_idx']['scene']
        indices = [i for i, x in enumerate(encoding) if x == scene_index]
        first_index = indices[0]
        second_index = indices[1]

        tree.append(token + '(' 
                    + str(current_idx - 1) + 
                    ',' + str(len(encoding) - first_index - 1) +
                    ',' + str(len(encoding) - second_index - 1) + 
                    ').')
        
        left_branch = regenerate_tree(
            [], encoding[1:first_index+1], current_idx - 1, vocab)
        middle_branch = regenerate_tree(
            [], encoding[first_index+1:second_index+1], current_idx - first_index - 1, vocab)
        right_branch = regenerate_tree(
            [], encoding[second_index+1:], current_idx - second_index - 1, vocab)
        
        tree.append(left_branch)
        tree.append(middle_branch)
        tree.append(right_branch)
    
    return tree


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(
            vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(
            vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(
            vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def load_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['3d_position'] = o['3d_coords']
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'],
                                           s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']

            if 'pixel_coords' in o:
                item['2d_coords'] = o['pixel_coords']

            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            table.append(item)
        scenes.append(table)
    return scenes


def load_embedding(path):
    return torch.Tensor(np.load(path))

def find_clevr_question_type(out_mod):
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
    if out_mod.startswith('between_projection'):
        q_type = 'between_projection'
    elif out_mod.startswith('between_bbox'):
        q_type = 'between_bbox'
    elif out_mod.startswith('between_proper'):
        q_type = 'between_proper'
    else:
        q_type = out_mod
    return q_type



stats = {
    'total': 0,
    'correct_ans': 0,
    'correct_prog': 0,
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
    'equal_shape': 0,
    'equal_shape_tot': 0,
    'equal_color': 0,
    'equal_color_tot': 0,
    'equal_material': 0,
    'equal_material_tot': 0,
    'equal_size': 0,
    'equal_size_tot': 0,
    'greater_than': 0,
    'greater_than_tot': 0,
    'less_than': 0,
    'less_than_tot': 0,
    'query_size': 0,
    'query_size_tot': 0,
    'query_color': 0,
    'query_color_tot': 0,
    'query_material': 0,
    'query_material_tot': 0,
    'query_shape': 0,
    'query_shape_tot': 0,
    'equal_integer': 0,
    'equal_integer_tot': 0,
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

class Context:

    def dist2(self, v, w):
        return (v[0] - w[0])**2 + (v[1] - w[1])**2

    def distToSegmentSquared(self, p, v, w):
        l2 = self.dist2(v, w)
        if l2 == 0:
            return self.dist2(p, v)
        t = ((p[0] - v[0]) * (w[0] - v[0]) +
             (p[1] - v[1]) * (w[1] - v[1])) / l2
        t = max(0, min(1, t))
        return self.dist2(p, [v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])])

    def distToSegment(self, p, v, w):
        p = [p.arguments[0].number, p.arguments[1].number]
        v = [v.arguments[0].number, v.arguments[1].number]
        w = [w.arguments[0].number, w.arguments[1].number]
        dist = math.sqrt(self.distToSegmentSquared(p, v, w))
        return clingo.Number(round(dist))
