""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores, top_scores


# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()

# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + 'models_final/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2


def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            context_label (int): Returns 0 if its same/similar context and 1 if out-of-context
    """
    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])
    scores_c1 = top_scores(score_c1)
    scores_c2 = top_scores(score_c2)
    top_bbox_c1, top_bbox_next_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2, top_bbox_next_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    bbox_overlap_next = is_bbox_overlap(top_bbox_next_c1, top_bbox_next_c2, iou_overlap_threshold)
    captions = v_data['caption1'] + v_data['caption2']
    
    if os.getenv("COSMOS_WORD_DISABLE") is None and (textual_sim >= textual_sim_threshold and (captions.find("hoax") != -1 or \
       captions.find("fake") != -1 or captions.find("claim") != -1 or captions.find("actual") != -1 or \
       captions.find("genuine") != -1 or captions.find("fabric") != -1 or captions.find("erroneous") != -1 or \
       captions.find("no evidence") != -1 or captions.find("satire") != -1 or \
       (v_data['caption1'].find("not") != -1 and v_data['caption2'].find("not") == -1) or \
       (v_data['caption1'].find("not") == -1 and v_data['caption2'].find("not") != -1))):
        context = 1
    else:
        #if bbox_overlap and (os.getenv("COSMOS_RECT_OPTIM") is None or \
        #   (top_bbox_c1[0] != 0 or top_bbox_c1[1] != 0 or float(scores_c1[0]) - float(scores_c1[1])) / abs(float(scores_c1[0])) > 0.1 or \
        #   (top_bbox_c2[0] != 0 or top_bbox_c2[1] != 0 or float(scores_c2[0]) - float(scores_c2[1])) / abs(float(scores_c2[0])) > 0.1 or bbox_overlap_next):
        if bbox_overlap:
            # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
            if textual_sim >= textual_sim_threshold:
                context = 0
            # Check for captions with different context : Same grounding with low textual overlap (Out of context)
            else:
                context = 1
        else:
            # Check for captions with same context : Different grounding (Not out of context)
            context = 0
    return scores_c1, scores_c2, context

if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
    ours_correct = 0
    lang_correct = 0

    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1
        scores_c1, scores_c2, pred_context = evaluate_context_with_bbox_overlap(v_data)
        
        if pred_context == actual_context:
            ours_correct += 1
            print("CORRECT:", pred_context, float(v_data['bert_base_score']) >= textual_sim_threshold, [float(i) for i in scores_c1], [float(i) for i in scores_c2])
        else:
            print("WRONG:", pred_context, float(v_data['bert_base_score']) >= textual_sim_threshold, [float(i) for i in scores_c1], [float(i) for i in scores_c2])

        if language_context == actual_context:
            lang_correct += 1

    print("Cosmos Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))
