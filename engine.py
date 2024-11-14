import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_cpu, use_optimizer
from metrics import MetronAtK

class Engine(object):
    """Meta Engine for training & evaluating NCF model"""

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config for real learning', str(config), 0)

        # Optimizer
        self.opt = use_optimizer(self.model, config)

        # Loss Function: BCELoss since we are dealing with implicit feedback
        self.crit = torch.nn.BCELoss()

        # Setup for CPU (since GPU isn't available)
        self.device = use_cpu()  # Force to use CPU
        self.model.to(self.device)  # Ensure model is on CPU

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)

        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_single_batch_mlp(self, users, items, ratings, item_category, media_type, channel_category, subscribers):
        assert hasattr(self, 'model'), 'Please specify the exact model!'

        # 장치 전송
        users = users.to(self.device)
        items = items.to(self.device)
        ratings = ratings.to(self.device)
        item_category = item_category.to(self.device)
        media_type = media_type.to(self.device)
        channel_category = channel_category.to(self.device)
        subscribers = subscribers.to(self.device)

        self.opt.zero_grad()
        ratings_pred = self.model(users, items, item_category, media_type, channel_category, subscribers)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            # Ensure we handle both user-item and item-user mappings
            user, item, rating = batch['user_id'], batch['item_id'], batch['target']
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print(f'[Training Epoch {epoch_id}] Batch {batch_id}, Loss {loss}')
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            test_users, test_items = test_users.to(self.device), test_items.to(self.device)
            negative_users, negative_items = negative_users.to(self.device), negative_items.to(self.device)

            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)

            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                negative_users.data.view(-1).tolist(),
                negative_items.data.view(-1).tolist(),
                negative_scores.data.view(-1).tolist()
            ]

        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print(f'[Evaluating Epoch {epoch_id}] HR = {hit_ratio:.4f}, NDCG = {ndcg:.4f}')
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)

    def prepare_data_for_evaluation(self, train_loader):
        """Prepare data for evaluation considering bi-directional recommendation."""
        all_users = []
        all_items = []
        all_ratings = []

        # Collecting data for bi-directional recommendations
        for batch in train_loader:
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            ratings = batch['target']

            # Add user-item pairs for evaluation (user to item)
            all_users.append(user_ids)
            all_items.append(item_ids)
            all_ratings.append(ratings)

            # Add item-user pairs for evaluation (item to user)
            all_users.append(item_ids)
            all_items.append(user_ids)
            all_ratings.append(ratings)

        # Concatenate for bi-directional evaluation
        users = torch.cat(all_users, dim=0).to(self.device)
        items = torch.cat(all_items, dim=0).to(self.device)
        ratings = torch.cat(all_ratings, dim=0).to(self.device)

        return users, items, ratings
