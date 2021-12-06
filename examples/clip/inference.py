'''
Author: jianzhnie
Date: 2021-12-03 17:49:30
LastEditTime: 2021-12-06 12:09:31
LastEditors: jianzhnie
Description:

'''
import sys

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from multimodal_transformers.models.clip.config_clip import ClipConfig as cfg
from multimodal_transformers.models.clip.datasets_clip import build_loaders, make_train_valid_dfs
from multimodal_transformers.models.clip.modeling_clip import CLIPModel
from tqdm import tqdm
from transformers import DistilBertTokenizer

sys.path.append('../../')


def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode='valid')

    model = CLIPModel().to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch['image'].to(cfg.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(cfg.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'])
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f'{cfg.image_path}/{match}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis('off')
    plt.savefig('demo.png')
    plt.show()


if __name__ == '__main__':
    model_file = 'best.pt'
    train_df, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(
        valid_df, model_path=model_file)
    find_matches(
        model,
        image_embeddings,
        query='one dog setting on the grass',
        image_filenames=valid_df['image'].values,
        n=9)
