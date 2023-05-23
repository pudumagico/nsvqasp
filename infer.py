import math
import json

import clingo
import torch
import torch.nn as nn
from pprint import pprint

from options import Options
from dataset import get_dataloader
from language.lstm.models.parser import Seq2seqParser

import tools
from tools import stats
from tools import Context

import time


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2)

opt = Options().parse()

if opt.ground_truth:
    scenes = json.load(open(opt.clevr_val_scene_path, 'r'))['scenes']

vocab = tools.load_vocab(opt.clevr_vocab_path)

yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.vision_weights)
lstm = Seq2seqParser(opt)
t = open(opt.theory)
theory = t.read()
t.close()

if opt.abduction:
    a = open(opt.abduction)
    abduction=a.read()
    a.close()
    abduction_stats = [] 

val_loader = get_dataloader(opt, 'val')

number_foil = 'foil(0).foil(1).foil(2).foil(3).foil(4).foil(5).foil(6).foil(7).foil(8).foil(9).foil(10).'
color_foil = 'foil(brown).foil(yellow).foil(green).foil(blue).foil(purple).foil(cyan).foil(red).foil(gray).'
size_foil = 'foil(small).foil(large).'
shape_foil = 'foil(sphere).foil(cylinder).foil(cube).'
material_foil = 'foil(metal).foil(rubber).'
bool_foil = 'foil(yes).foil(no).'

pos_examples = []

for src, tgt, ans, img_idx, img_filenames in val_loader:
    batch_start = time.time()

    lstm.set_input(src, tgt)
    pred_programs = lstm.parse()

    path_img_filenames = []
    for filename in img_filenames:
        path_img_filenames.append(f'{opt.clevr_val_image_path}/{filename}')
    abstract_scenes = yolo(path_img_filenames)

    for i, pred_program in enumerate(pred_programs):

        img_filename = img_filenames[i]
        pred_ans = None
        
        gt_ans = vocab['answer_idx_to_token'][ans[i].item()]
        if opt.ground_truth:
            gt_scene = None
            for scene in scenes:
                if scene['image_filename'] == img_filename:
                    gt_scene = tools.gt_scene(scene)
                    break
            gt_question = tools.language2facts(tgt[i][1:], vocab)

        img = path_img_filenames[i]
        q_type = tools.find_clevr_question_type(
            vocab['program_idx_to_token'][tgt[i][1].item()])
        # print(abstract_scenes.pandas().xyxy[i])
        abstract_scene = abstract_scenes.pandas().xyxy[i].values.tolist()
        scene = tools.vision2facts(abstract_scene)
        question = tools.language2facts(pred_program, vocab)

        ctl = clingo.Control(['--warn=none'])
        ctl.add("base", [], theory)
        ctl.add("base", [], scene)
        ctl.add("base", [], question)
        ctl.ground([("base", [])], context=Context())

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                pred_ans = str(model.symbols(shown=True)[0].arguments[0])
        
        # print(img_filename)
        # print(abstract_scene)
        # print(scene)
        # print(question)
        # print(pred_ans)
        # print(gt_ans)

        if pred_ans == gt_ans:
            stats[q_type] += 1
            stats['correct_ans'] += 1
            
            pos_examples.append([question, scene, pred_ans, f'foil({pred_ans})'])
            # print('------------------')
            # pprint({'img_filename':img_filename,
            # 'scene':scene,
            # 'question':question,
            # 'pred_ans':pred_ans,
            # 'gt_scene':gt_scene,
            # 'gt_question':gt_question,
            # 'gt_ans':gt_ans,
            # # 'model':str(model)
            # })
        # else:
        #     print('---------')
        #     pprint({'img_filename':img_filename,
        #             'scene':scene,
        #             'question':question,
        #             'pred_ans':pred_ans,
        #             'gt_scene':gt_scene,
        #             'gt_question':gt_question,
        #             'gt_ans':gt_ans,
                    
        #             })
            # print(scene)
            # print(question)
            # print(pred_ans)
            # print(gt_scene)
            # print(gt_question)
            # print(gt_ans)
        elif opt.abduction:
            ta = open('./reasoning/theory_abduction.lp')
            ta=ta.read()
            
        
            foil_ans = f'foil({gt_ans}).'
            if foil_ans in number_foil:
                foil = number_foil.replace(foil_ans, '')
            elif foil_ans in color_foil:
                foil = color_foil.replace(foil_ans, '')
            elif foil_ans in material_foil:
                foil = material_foil.replace(foil_ans, '')
            elif foil_ans in shape_foil:
                foil = shape_foil.replace(foil_ans, '')
            elif foil_ans in size_foil:
                foil = size_foil.replace(foil_ans, '')
            else:
                foil = bool_foil.replace(foil_ans, '')
            
        
            abduction_stat = {'img_filename': img_filename, 
                              'models': [], 'g_time':0, 's_time':0, 
                              'question': question, 'scene': scene, 'costs': [],
                              'foil': foil, 'pred_ans': pred_ans}
            
            ctl_abduction = clingo.Control(['--warn=none', '--opt-strategy=usc'])
            
            ctl_abduction.add("base", [], abduction)
            ctl_abduction.add("base", [], ta)
            ctl_abduction.add("base", [], scene)
            ctl_abduction.add("base", [], question)
            ctl_abduction.add("base", [], foil)
            ctl_abduction.assign_external(clingo.Function('e_id'), truth=True)
            ctl_abduction.assign_external(clingo.Function('e_position'), truth=True)
            ctl_abduction.assign_external(clingo.Function('e_size'), truth=True)
            ctl_abduction.assign_external(clingo.Function('e_color'), truth=True)
            ctl_abduction.assign_external(clingo.Function('e_material'), truth=True)
            ctl_abduction.assign_external(clingo.Function('e_shape'), truth=True)
            
            ctl_abduction.ground([("base", [])], context=Context())
            with ctl_abduction.solve(yield_=True) as handle:
                for model in handle:
                    # abduction_stat['models'].append(str(model).split(' '),)
                    # abduction_stat['costs'].append(int(model.cost[0]))
                    pprint({'img_filename':img_filename,
                    'scene':scene,
                    'question':question,
                    'pred_ans':pred_ans,
                    'gt_scene':gt_scene,
                    'gt_question':gt_question,
                    'gt_ans':gt_ans,
                    'model':str(model),
                    'foil': foil
                    })
                    print('------------------')
                    

        if question == gt_question:
            stats['correct_prog'] += 1
        stats['%s_tot' % q_type] += 1
        stats['total'] += 1
    batch_end = time.time()
    pose_file = open('positive_examples.txt', 'w')
    json.dump(pos_examples, pose_file)
    pose_file.close()
    print(stats)


    # print(stats['correct_ans'], stats['total'], batch_end - batch_start)
    # abduction_file = open('abduction_single_foil.txt', 'w')
    # json.dump(abduction_stats, abduction_file)
    # abduction_file.close()
    # stats_file = open('stats.txt', 'w')
    # json.dump(stats, stats_file)
    # stats_file.close()
    # pprint(stats)

