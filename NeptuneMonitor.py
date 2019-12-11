import neptune
from tensorflow.keras.callbacks import BaseLogger


class NeptuneMonitor(BaseLogger):
    def __init__(self, name, api_token, prj_name, params: tuple = None):
        assert api_token is not None
        assert prj_name is not None
        super(BaseLogger, self).__init__()
        self.my_name = name
        self.stateful_metrics = set(['loss'] or [])

        neptune.init(
            api_token=api_token,
            project_qualified_name=prj_name)
        self.experiment = neptune.create_experiment(name=self.my_name, params=params)
        self.log_neptune = True

    def on_epoch_end(self, epoch, logs={}):
        #acc = logs['acc']
        loss = logs['loss']

        if self.log_neptune:
            self.experiment.append_tag(self.my_name)
            #self.experiment.send_metric('acc', acc)
            self.experiment.send_metric('loss', loss)
            #self.experiment.send_metric('epoch', epoch)
            
    def on_train_end(self, logs={}):
        if self.log_neptune:
            self.experiment.stop()