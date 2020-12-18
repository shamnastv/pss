import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnn import DynamicLSTM


def get_features(inputs, device, initial):
    feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target = [], [], [], [], [], [], []
    a_mask, a_value = [], []
    for d in inputs:
        feature_ids.append(d['word_ids'])
        aspect_ids.append(d['target_ids'])
        feature_lens.append(d['word_count'])
        aspect_lens.append(d['target_word_count'])
        position_weight.append(d['position_weight'])
        masks.append(d['mask'])
        target.append(d['y'])
        if not initial:
            a_mask.append(d['amask'])
            a_value.append(d['avalue'])

    feature_ids = torch.tensor(feature_ids).long().to(device)
    aspect_ids = torch.tensor(aspect_ids).long().to(device)
    feature_lens = torch.tensor(feature_lens).long().to(device)
    aspect_lens = torch.tensor(aspect_lens).long().to(device)
    position_weight = torch.tensor(position_weight).float().to(device)
    masks = torch.tensor(masks).float().to(device)
    masks = masks.eq(0)
    target = torch.tensor(target).long().to(device)
    if not initial:
        a_mask = torch.tensor(a_mask).float().to(device)
        a_value = torch.tensor(a_value).float().to(device)
    else:
        a_mask = None
        a_value = None

    return feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target, a_mask, a_value


class Model(nn.Module):
    def __init__(self, args, num_clasees, word_embeddings, device):
        super(Model, self).__init__()
        self.device = device

        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float().to(device))

        self.lstm1 = DynamicLSTM(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm3 = DynamicLSTM(2 * args.hidden_dim, args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(4 * args.hidden_dim, 2 * args.hidden_dim)
        self.fc = nn.Linear(2 * args.hidden_dim, num_clasees)
        self.dropout = nn.Dropout(args.dropout)
        self.linear2 = nn.Linear(2 * args.hidden_dim, 1)

    def forward(self, inputs, initial):
        feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target, a_mask, a_value\
            = get_features(inputs, self.device, initial)

        features = self.word_embeddings(feature_ids)
        aspects = self.word_embeddings(aspect_ids)
        v, (_, _) = self.lstm1(features, feature_lens)
        e, (_, _) = self.lstm2(aspects, aspect_lens)

        v = self.dropout(v.transpose(1, 2))
        e = self.dropout(e.transpose(1, 2))
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            aspect_mid = self.dropout(aspect_mid)
            t = torch.sigmoid(self.linear2(v.transpose(1, 2))).transpose(1, 2)
            v = (1 - t) * aspect_mid + t * v
            v = position_weight.unsqueeze(2) * v.transpose(1, 2)
            v = v.transpose(1, 2)

        v = v.transpose(1, 2)
        # z, (_, _) = self.lstm3(v, feature_lens)
        # query = torch.max(e, dim=2)[0].unsqueeze(2)
        hidden_fwd, hidden_bwd = e.chunk(2, 1)
        query = torch.cat((hidden_fwd[:, :, -1], hidden_bwd[:, :, 0]), dim=1).unsqueeze(2)

        alpha = torch.bmm(v, query)
        alpha.masked_fill_(masks.unsqueeze(2), -1e9)
        alpha = F.softmax(alpha, 1)
        z = torch.bmm(alpha.transpose(1, 2), v)
        z = self.fc(z.squeeze(1))
        return z, alpha.squeeze(2), target, a_mask, a_value
