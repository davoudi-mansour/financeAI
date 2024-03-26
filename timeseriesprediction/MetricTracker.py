class MetricTracker:
    def __init__(self,metrics=['loss']):
        self.metrics = metrics
        self.train_loss = {k:[] for k in metrics}
        self.val_loss = {k:[] for k in metrics}
        self.test_loss = {k:[] for k in metrics}

    def log(self,train_log,val_log,test_log):
        for metric in self.metrics:
            self.train_loss[metric].append(train_log[metric])
            self.val_loss[metric].append(val_log[metric])
            self.test_loss[metric].append(test_log[metric])