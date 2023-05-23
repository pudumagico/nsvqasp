# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import json
import os
import math
import random
from collections import defaultdict, Counter

import numpy as np
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
"""
Utilities for working with function program representations of questions.

Some of the metadata about what question node types are available etc are stored
in a JSON metadata file.
"""


# Handlers for answering questions. Each handler receives the scene structure
# that was output from Blender, the node, and a list of values that were output
# from each of the node's inputs; the handler should return the computed output
# value from this node.

def point_to_line_dist(point, line):

    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) + 
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist

import math

def dist2(v, w):
    return (v[0] - w[0])**2 + (v[1] - w[1])**2

def distToSegmentSquared (p, v, w):
    l2 = dist2(v, w)
    if l2 == 0:
        return dist2(p, v)
    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
    t = max(0, min(1, t))
    return dist2(p, [ v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]) ])

def distToSegment(p, v, w):
      return math.sqrt(distToSegmentSquared(p, v, w))


def scene_handler(scene_struct, inputs, side_inputs):
    # Just return all objects in the scene
    # print(scene_struct['objects'])
    return list(range(len(scene_struct['objects'])))


def scene_categorical_handler(scene_struct, inputs, side_inputs):
    # Return a random int between zero and the sum of objects in the scene
    return int(side_inputs[0])
    # return random.randint(len(scene_struct['objects']))

def all_different_handler(scene_struct, inputs, side_inputs):
    ans = True

    objects = scene_struct['objects']
    obj_attrs = []

    # diff_objs = []
    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])


    for attrs in obj_attrs:
        if obj_attrs.count(attrs) > 1:
            return False
    # ans = len(diff_objs)

    return ans

def two_equal_handler(scene_struct, inputs, side_inputs):

    ans = False
    objects = scene_struct['objects'].copy()
    obj_attrs = []

    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])

    
    for obj in obj_attrs:
        if obj_attrs.count(obj) == 2:
            ans = True
    return ans

def exactly_two_equal_handler(scene_struct, inputs, side_inputs):

    ans = False
    objects = scene_struct['objects'].copy()
    count = 0
    obj_attrs = []

    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])


    for obj in obj_attrs:
        if obj_attrs.count(obj) == 2:
            count+=1
            # ans = True
    if count == 2:
        ans = True
    return ans

def three_equal_handler(scene_struct, inputs, side_inputs):

    ans = False
    objects = scene_struct['objects'].copy()
    obj_attrs = []
    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])

  
    for obj in obj_attrs:
        if obj_attrs.count(obj) == 3:
            ans = True 
    return ans

def exactly_three_equal_handler(scene_struct, inputs, side_inputs):
    # print(scene_struct)

    ans = False
    count = 0
    objects = scene_struct['objects'].copy()
    obj_attrs = []
    # print(objects)
    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])

    for obj in obj_attrs:
        if obj_attrs.count(obj) == 3:
            count += 1

    if count == 3:
        ans = True
    return ans

def four_equal_handler(scene_struct, inputs, side_inputs):

    ans = False
    objects = scene_struct['objects'].copy()
    obj_attrs = []

    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])

    
    for obj in obj_attrs:
        if obj_attrs.count(obj) == 4:
            ans = True 
    return ans

def exactly_four_equal_handler(scene_struct, inputs, side_inputs):

    ans = False
    count = 0


    obj_attrs = []
    
    objects = scene_struct['objects'].copy()

    for obj in objects:
        obj_attrs.append([obj['material'], obj['size'], obj['color'], obj['shape']])


    for obj in obj_attrs:
        if obj_attrs.count(obj) == 4:
            count += 1
    if count == 4:
        ans = True
    
    return ans


def between_projection_handler(scene_struct, inputs, side_inputs):
    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])
    obj3_px = np.array(scene_struct['objects'][inputs[2]]['pixel_coords'])

    if obj1_px[0] < obj3_px[0] < obj2_px[0] or obj2_px[0] < obj3_px[0] < obj1_px[0]:
        return True
    else:
        return False

def between_bbox_handler(scene_struct, inputs, side_inputs):
    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])
    obj3_px = np.array(scene_struct['objects'][inputs[2]]['pixel_coords'])

    if (obj1_px[0] < obj3_px[0] < obj2_px[0] or obj2_px[0] < obj3_px[0] < obj1_px[0]) and \
        (obj1_px[1] < obj3_px[1] < obj2_px[1] or obj2_px[1] < obj3_px[1] < obj1_px[1]):
        return True
    else:
        return False

def between_proper_handler(scene_struct, inputs, side_inputs):

    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])
    obj3_px = np.array(scene_struct['objects'][inputs[2]]['pixel_coords'])
    delta = 0.5
    x = distToSegment(obj3_px, obj1_px, obj2_px)
    if x < delta:
        return True
    else:
        return False

def boolean_negation_handler(scene_struct, inputs, side_inputs):
    return not inputs[0]


def count_between_projection_handler(scene_struct, inputs, side_inputs):
    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])

    count = 0
    
    for obj in scene_struct['objects']:
        obj3_px = obj['pixel_coords']
        if obj1_px[0] < obj3_px[0] < obj2_px[0] or obj2_px[0] < obj3_px[0] < obj1_px[0]:
            count += 1

    return count


def count_between_bbox_handler(scene_struct, inputs, side_inputs):
    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])

    count = 0

    for obj in scene_struct['objects']:
        obj3_px = obj['pixel_coords']
        if (obj1_px[0] < obj3_px[0] < obj2_px[0] or obj2_px[0] < obj3_px[0] < obj1_px[0]) and \
            (obj1_px[1] < obj3_px[1] < obj2_px[1] or obj2_px[1] < obj3_px[1] < obj1_px[1]):
            count += 1
    return count


def count_between_proper_handler(scene_struct, inputs, side_inputs):
    # Return a random int between zero and the sum of objects in the scene
    
    obj1_px = np.array(scene_struct['objects'][inputs[0]]['pixel_coords'])
    obj2_px = np.array(scene_struct['objects'][inputs[1]]['pixel_coords'])
    
    delta = 0.5
    count = 0

    for i in range(len(scene_struct['objects'])):
        if i in inputs:
            pass
        else:
            obj3_px = np.array(scene_struct['objects'][i]['pixel_coords'])
            x = distToSegment(obj3_px, obj1_px, obj2_px)

            if x < delta:
                count += 1

    return count

def set_difference_handler(scene_struct, inputs, side_inputs):
    return len(scene_struct['objects']) - inputs[0]


def count_different_handler(scene_struct, inputs, side_inputs):
    ans = True

    objects = scene_struct['objects']
    print(objects)
    diff_objs = []

    for obj in objects:
        if obj not in diff_objs:
            diff_objs.append(obj)
    ans = len(diff_objs)

    return ans

def make_filter_handler(attribute):
    def filter_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        output = []
        for idx in inputs[0]:
            atr = scene_struct['objects'][idx][attribute]
            if value == atr or value in atr:
                output.append(idx)
        return output
    return filter_handler


def unique_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]


def vg_relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    output = set()
    for rel in scene_struct['relationships']:
        if rel['predicate'] == side_inputs[0] and rel['subject_idx'] == inputs[0]:
            output.add(rel['object_idx'])
    return sorted(list(output))


def relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]
    return scene_struct['relationships'][relation][inputs[0]]


def relate_multiple_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]
    for i in inputs[0]:
        return scene_struct['relationships'][relation][inputs[0][i]]


def union_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) | set(inputs[1])))


def intersect_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) & set(inputs[1])))


def not_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    if isinstance(inputs[1], int):
        inputs[1] = [inputs[1]]
    return sorted(list(set(inputs[0]) - set(inputs[1])))


def count_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    return len(inputs[0])


def make_same_attr_handler(attribute):
    def same_attr_handler(scene_struct, inputs, side_inputs):
        cache_key = '_same_%s' % attribute
        if cache_key not in scene_struct:
            cache = {}
            for i, obj1 in enumerate(scene_struct['objects']):
                same = []
                for j, obj2 in enumerate(scene_struct['objects']):
                    if i != j and obj1[attribute] == obj2[attribute]:
                        same.append(j)
                cache[i] = same
            scene_struct[cache_key] = cache

        cache = scene_struct[cache_key]
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        return cache[inputs[0]]
    return same_attr_handler


def make_query_handler(attribute):
    def query_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        idx = inputs[0]
        obj = scene_struct['objects'][idx]
        assert attribute in obj
        val = obj[attribute]
        if type(val) == list and len(val) != 1:
            return '__INVALID__'
        elif type(val) == list and len(val) == 1:
            return val[0]
        else:
            return val
    return query_handler


def exist_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return len(inputs[0]) > 0


def equal_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] == inputs[1]


def less_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] > inputs[1]


# Register all of the answering handlers here.
# TODO maybe this would be cleaner with a function decorator that takes
# care of registration? Not sure. Also what if we want to reuse the same engine
# for different sets of node types?
execute_handlers = {
    'scene': scene_handler,
    'scene_categorical': scene_categorical_handler,
    'filter_color': make_filter_handler('color'),
    'filter_shape': make_filter_handler('shape'),
    'filter_material': make_filter_handler('material'),
    'filter_size': make_filter_handler('size'),
    'filter_objectcategory': make_filter_handler('objectcategory'),
    'unique': unique_handler,
    'relate': relate_handler,
    'relate_multiple': relate_multiple_handler,
    'union': union_handler,
    'intersect': intersect_handler,
    'not': not_handler,
    'count': count_handler,
    'query_color': make_query_handler('color'),
    'query_shape': make_query_handler('shape'),
    'query_material': make_query_handler('material'),
    'query_size': make_query_handler('size'),
    'exist': exist_handler,
    'equal_color': equal_handler,
    'equal_shape': equal_handler,
    'equal_integer': equal_handler,
    'equal_material': equal_handler,
    'equal_size': equal_handler,
    'equal_object': equal_handler,
    'less_than': less_than_handler,
    'greater_than': greater_than_handler,
    'same_color': make_same_attr_handler('color'),
    'same_shape': make_same_attr_handler('shape'),
    'same_size': make_same_attr_handler('size'),
    'same_material': make_same_attr_handler('material'),
    'between_projection': between_projection_handler,
    'between_bbox': between_bbox_handler,
    'between_proper': between_proper_handler,
    'boolean_negation': boolean_negation_handler,
    'two_equal': two_equal_handler,
    'three_equal': three_equal_handler,
    'four_equal': four_equal_handler,
    'exactly_two_equal': exactly_two_equal_handler,
    'exactly_three_equal': exactly_three_equal_handler,
    'exactly_four_equal': exactly_four_equal_handler,
    'all_different': all_different_handler,
    'count_between_projection': count_between_projection_handler,
    'count_between_bbox': count_between_bbox_handler,
    'count_between_proper': count_between_proper_handler,
    'count_different': count_different_handler,
    'set_difference': set_difference_handler
}


def answer_question(question, metadata, scene_struct, all_outputs=False,
                    cache_outputs=True):
    """
    Use structured scene information to answer a structured question. Most of the
    heavy lifting is done by the execute handlers defined above.

    We cache node outputs in the node itself; this gives a nontrivial speedup
    when we want to answer many questions that share nodes on the same scene
    (such as during question-generation DFS). This will NOT work if the same
    nodes are executed on different scenes.
    """
    all_input_types, all_output_types = [], []
    node_outputs = []
    for node in question['nodes']:
        if cache_outputs and '_output' in node:
            node_output = node['_output']
        else:
            node_type = node['type']
            msg = 'Could not find handler for "%s"' % node_type
            assert node_type in execute_handlers, msg
            handler = execute_handlers[node_type]
            node_inputs = [node_outputs[idx] for idx in node['inputs']]
            side_inputs = node.get('side_inputs', [])
            node_output = handler(scene_struct, node_inputs, side_inputs)
            if cache_outputs:
                node['_output'] = node_output
        node_outputs.append(node_output)
        if node_output == '__INVALID__':
            break

    if all_outputs:
        return node_outputs
    else:
        return node_outputs[-1]


def insert_scene_node(nodes, idx):
    # First make a shallow-ish copy of the input
    new_nodes = []
    for node in nodes:
        new_node = {
            'type': node['type'],
            'inputs': node['inputs'],
        }
        if 'side_inputs' in node:
            new_node['side_inputs'] = node['side_inputs']
        new_nodes.append(new_node)

    # Replace the specified index with a scene node
    new_nodes[idx] = {'type': 'scene', 'inputs': []}

    # Search backwards from the last node to see which nodes are actually used
    output_used = [False] * len(new_nodes)
    idxs_to_check = [len(new_nodes) - 1]
    while idxs_to_check:
        cur_idx = idxs_to_check.pop()
        output_used[cur_idx] = True
        idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

    # Iterate through nodes, keeping only those whose output is used;
    # at the same time build up a mapping from old idxs to new idxs
    old_idx_to_new_idx = {}
    new_nodes_trimmed = []
    for old_idx, node in enumerate(new_nodes):
        if output_used[old_idx]:
            new_idx = len(new_nodes_trimmed)
            new_nodes_trimmed.append(node)
            old_idx_to_new_idx[old_idx] = new_idx

    # Finally go through the list of trimmed nodes and change the inputs
    for node in new_nodes_trimmed:
        new_inputs = []
        for old_idx in node['inputs']:
            new_inputs.append(old_idx_to_new_idx[old_idx])
        node['inputs'] = new_inputs

    return new_nodes_trimmed


def is_degenerate(question, metadata, scene_struct, answer=None, verbose=False):
    """
    A question is degenerate if replacing any of its relate nodes with a scene
    node results in a question with the same answer.
    """
    if answer is None:
        answer = answer_question(question, metadata, scene_struct)

    for idx, node in enumerate(question['nodes']):
        if node['type'] == 'relate':
            new_question = {
                'nodes': insert_scene_node(question['nodes'], idx)
            }
            new_answer = answer_question(new_question, metadata, scene_struct)
            if verbose:
                print('here is truncated question:')
                for i, n in enumerate(new_question['nodes']):
                    name = n['type']
                    if 'side_inputs' in n:
                        name = '%s[%s]' % (name, n['side_inputs'][0])
                    print(i, name, n['_output'])
                print('new answer is: ', new_answer)

            if new_answer == answer:
                return True

    return False
