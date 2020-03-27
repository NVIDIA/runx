import os
import shutil
import tempfile
import torch
import unittest

from parameterized import parameterized

import logx


class LogXTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.log_x = logx.LogX()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialize(self):
        self.log_x.initialize(self.test_dir, coolname=True, tensorboard=True)

    @parameterized.expand([
        (rank, use_tensorboard)
        for rank in range(8)
        for use_tensorboard in [True, False]
    ])
    def test_tensorboard(self, rank, use_tensorboard):
        self.log_x.initialize(self.test_dir, tensorboard=use_tensorboard,
                              global_rank=rank,
                              eager_flush=True)

        self.log_x.add_scalar('train/loss', 42, 1)

        found_events = False

        for f in os.listdir(self.test_dir):
            if f.startswith('events.out.tfevents.'):
                found_events = True
                break

        self.assertEqual(use_tensorboard and rank == 0, found_events)

    @parameterized.expand([
        [True],
        [False],
    ])
    def test_tb_flushing(self, eager_flush):
        old_flush = self.log_x._flush_tensorboard

        flushed = [False]

        def mock_flush():
            flushed[0] = True
            old_flush()

        self.log_x._flush_tensorboard = mock_flush

        self.log_x.initialize(self.test_dir, tensorboard=True,
                              eager_flush=eager_flush)

        # This ensures that `eager_flush` is honored, even when calling through
        # to the actual tensorboard API
        self.log_x.tensorboard.add_scalars('train/vals', {
            'val1': 10,
            'val2': 20,
        }, 1)

        # This will always be called, regardless of what `eager_flush` is set
        # to. The actual function will check that value before actually calling
        # flush.
        self.assertTrue(flushed[0])

    @parameterized.expand([
        [True],
        [False],
    ])
    def test_tb_suspend_flush(self, flush_at_end):
        flushed = [False]

        def mock_flush():
            if self.log_x.eager_flush:
                flushed[0] = True

        self.log_x._flush_tensorboard = mock_flush

        self.log_x.initialize(self.test_dir, tensorboard=True,
                              eager_flush=True)

        self.assertTrue(self.log_x.eager_flush)

        with self.log_x.suspend_flush(flush_at_end):
            self.assertFalse(self.log_x.eager_flush)

            for i in range(10):
                self.log_x.tensorboard.add_scalar('train/loss', 9 - i, i)

            self.assertFalse(flushed[0])

        self.assertTrue(self.log_x.eager_flush)
        self.assertEqual(flushed[0], flush_at_end)

    @parameterized.expand([
        (phase, rank, epoch)
        for phase in ['train', 'val']
        for rank in range(2)
        for epoch in [None, 3]
    ])
    def test_metrics(self, phase, rank, epoch):
        self.log_x.initialize(self.test_dir, tensorboard=True,
                              global_rank=rank)

        metrics = [
            {'top1': 0.85, 'top5': 0.91, 'auc': 0.89},
            {'top1': 0.855, 'top5': 0.92, 'auc': 0.895},
        ]
        epochs = [
            epoch + i if epoch is not None else i
            for i in range(len(metrics))
        ]

        for e, metric in zip(epochs, metrics):
            self.log_x.metric(
                phase,
                epoch=e if epoch is not None else None,
                metrics=metric)

        # Force all of the writers to flush
        del self.log_x

        metrics_file = os.path.join(self.test_dir, 'metrics.csv')

        if rank == 0:
            self.assertTrue(os.path.exists(metrics_file))

            with open(metrics_file, 'r') as fd:
                lines = [line.strip() for i, line in enumerate(fd.readlines())
                         if i > 0]
                lines = [line for line in lines if len(line) > 0]
                if phase == 'train':
                    self.assertEqual(len(lines), 0)
                elif phase == 'val':
                    self.assertEqual(len(lines), len(metrics))
                    for line, expected_epoch, expected_metric in \
                            zip(lines, epochs, metrics):
                        vals = line.split(',')
                        counter = 0
                        self.assertEqual(vals[counter], 'val')
                        counter += 1
                        for k, v in expected_metric.items():
                            self.assertEqual(vals[counter], k)
                            self.assertEqual(vals[counter + 1], str(v))
                            counter += 2
                        self.assertEqual(vals[counter], 'epoch')
                        counter += 1
                        self.assertEqual(int(vals[counter]), expected_epoch)
        else:
            self.assertFalse(os.path.exists(metrics_file))

    def test_best_checkpoint(self):
        self.log_x.initialize(self.test_dir)

        self.assertIsNone(self.log_x.get_best_checkpoint())

        best_path = os.path.join(self.test_dir, 'best_checkpoint_ep5.pth')

        with open(best_path, 'w') as fd:
            fd.write('hello')

        self.assertEqual(self.log_x.get_best_checkpoint(), best_path)

        best_path = os.path.join(self.test_dir, 'best_checkpoint_ep10.pth')

        with open(best_path, 'w') as fd:
            fd.write('hello2')

        self.assertEqual(self.log_x.get_best_checkpoint(), best_path)

        # This shouldn't change what the best path is due to the epoch rule
        with open(os.path.join(self.test_dir, 'best_checkpoint_ep1.pth'),
                  'w') as fd:
            fd.write('hello3')

        self.assertEqual(self.log_x.get_best_checkpoint(), best_path)

    @parameterized.expand([
        [0],
        [1],
    ])
    def test_save_model(self, rank):
        self.log_x.initialize(self.test_dir, global_rank=rank)

        model1 = {
            'val1': 42,
            'val2': 44,
            'val3': torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        }

        self.log_x.save_model(model1, metric=0.5, epoch=0)

        self.assertEqual(
            os.path.exists(
                os.path.join(self.test_dir, 'best_checkpoint_ep0.pth')),
            rank == 0)
        self.assertEqual(
            os.path.exists(
                os.path.join(self.test_dir, 'last_checkpoint_ep0.pth')),
            rank == 0)

        def dict_test(d1, d2):
            # d2 is allowed to be a superset of d1
            for k, v in d1.items():
                self.assertIn(k, d2)

                if torch.is_tensor(v):
                    self.assertTrue(torch.allclose(v, d2[k]))
                else:
                    self.assertEqual(v, d2[k])

        if rank == 0:
            dict_test(model1, torch.load(self.log_x.get_best_checkpoint()))

        model2 = {
            'val1': 47,
            'val2': 50,
            'val3': torch.tensor([[5, 6], [7, 8], [9, 0]], dtype=torch.float32)
        }

        self.log_x.save_model(model2, metric=0.7, epoch=50)

        self.assertFalse(
            os.path.exists(
                os.path.join(self.test_dir, 'best_checkpoint_ep0.pth')))
        self.assertEqual(
            os.path.exists(
                os.path.join(self.test_dir, 'best_checkpoint_ep50.pth')),
            rank == 0)
        self.assertEqual(
            os.path.exists(
                os.path.join(self.test_dir, 'last_checkpoint_ep50.pth')),
            rank == 0)

        if rank == 0:
            dict_test(model2, torch.load(self.log_x.get_best_checkpoint()))

        model3 = {
            'val1': 2,
            'val2': 3,
            'val3': torch.rand(3, 3, 3, dtype=torch.float32),
        }

        # This metric is worse than the previous, so it shouldn't replace best
        self.log_x.save_model(model3, metric=0.6, epoch=60)

        self.assertEqual(
            os.path.exists(
                os.path.join(self.test_dir, 'last_checkpoint_ep60.pth')),
            rank == 0)

        if rank == 0:
            dict_test(model2, torch.load(self.log_x.get_best_checkpoint()))

        # Now, the hard part. Kill this log_x, and create a new one, to verify
        # that resumption works
        del self.log_x
        self.log_x = logx.LogX()
        self.log_x.initialize(self.test_dir, global_rank=rank)

        # Again, the best checkpoint should still be model2 at epoch 50
        self.log_x.save_model(model3, metric=0.6, epoch=60)

        if rank == 0:
            dict_test(model2, torch.load(self.log_x.get_best_checkpoint()))


if __name__ == '__main__':
    unittest.main()
