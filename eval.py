import argparse
import json
import os

from pandas.io.json import json_normalize

from tools.cocoeval import COCOScorer, suppress_stdout_stderr


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append({
                'image_id': row[1],
                'cap_id': len(gts[row[1]]),
                'caption': row[0]
            })
        else:
            gts[row[1]] = []
            gts[row[1]].append({
                'image_id': row[1],
                'cap_id': len(gts[row[1]]),
                'caption': row[0]
            })
    return gts


def main(opt):
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["videoinfo_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    samples = {}
    video_ids = open(opt['video_ids'])
    sents = open(opt['pred'])
    for video_id in video_ids:
        # strip file extensions
        video_id = video_id.split('.')[0]
        sent = sents.readline().strip()
        samples[video_id] = [{'image_id': video_id, 'caption': sent}]
    video_ids.close()
    sents.close()
    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    print(valid_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-videoinfo_json', type=str, default='data/videodatainfo_2017.json')
    parser.add_argument(
        '-video_ids',
        type=str,
        help='file containing video ids corresponding to pred')
    parser.add_argument(
        '-pred', type=str, help='pred.txt produced by translate.py')

    args = parser.parse_args()
    args = vars(args)
    main(args)
