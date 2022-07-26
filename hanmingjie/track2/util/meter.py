
class Meter:
    def __init__(self, metric=None, show=True):
        self.show = show
        if metric == None:
            self.f1 = 0
            self.f1_cls = 0
            self.precision = 0
            self.precision_cls = 0
            self.recall = 0
            self.recall_cls = 0
            self.accuracy = 0
        else:
            self.f1 = metric['f1']
            self.f1_cls = metric['f1_cls']
            self.precision = metric['precision']
            self.precision_cls = metric['precision_cls']
            self.recall = metric['recall']
            self.recall_cls = metric['recall_cls']
            self.accuracy = metric['accuracy']

    @classmethod
    def set_category_num(cls, category_num):
        cls.CATEGORY_NUM = category_num

    def update(self, metric):
        self.f1 = metric['f1']
        self.f1_cls = metric['f1_cls']
        self.precision = metric['precision']
        self.precision_cls = metric['precision_cls']
        self.recall = metric['recall']
        self.recall_cls = metric['recall_cls']
        self.accuracy = metric['accuracy']

    def __str__(self) -> str:
        if self.show:
            show_str = 'f1:{:.4f}(avg), '.format(self.f1)
            for i in range(self.CATEGORY_NUM):
                show_str += '{:.4f}({}), '.format(self.f1_cls[i], i)
            show_str += 'precision:{:.4f}(avg), '.format(self.precision)
            for i in range(self.CATEGORY_NUM):
                show_str += '{:.4f}({}), '.format(self.precision_cls[i], i)
            show_str += 'recall:{:.4f}(avg), '.format(self.recall)
            for i in range(self.CATEGORY_NUM):
                show_str += '{:.4f}({}), '.format(self.recall_cls[i], i)
            show_str += 'accuracy:{:.4f}'.format(self.accuracy)

            return show_str
        else:
            return 'no metric'
    