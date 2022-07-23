import time 

class Timer:
    # 'all' or 'epoch'
    timer_type = None
    start_time = None

    @classmethod
    def set_up(cls, timer_type='all'):
        cls.timer_type = timer_type
        cls.start_time = time.time()

    @classmethod
    def clear(cls):
        cls.start_time = time.time()

    @classmethod
    def calculate_remain(cls, spend_epoch, spend_iteration, max_epoch, max_iteration, resume_epoch=None):
        curr_time = time.time()
        start_time = cls.start_time
        if resume_epoch is not None:
            spend_epoch -= resume_epoch
            max_epoch -= resume_epoch
        if cls.timer_type == 'all':
            spend_iter_all = (spend_epoch - 1) * max_iteration + spend_iteration
            max_iter_all = max_epoch * max_iteration
            remain_time = (curr_time - start_time) / spend_iter_all * max_iter_all - (curr_time - start_time)
        elif cls.timer_type == 'epoch':
            remain_time = (curr_time - start_time) / spend_iteration * max_iteration - (curr_time - start_time)
        return cls.time_transform(remain_time)

    @classmethod
    def calculate_spend(cls):
        curr_time = time.time()
        start_time = cls.start_time
        return cls.time_transform(curr_time - start_time)


    @classmethod
    def time_transform(cls, time):
        if time < 60:
            second = int(time)
            return str(second) + 's'
        elif time < 60*60:
            minute = int(time) // 60
            second = int(time) - minute * 60
            return str(minute) + 'm ' + str(second) + 's'
        elif time < 60*60*24:
            hour = int(time) // (60*60)
            minute = (int(time) - hour * (60*60)) // 60
            second = int(time) - hour * (60*60) - minute * 60
            return str(hour) + 'h ' + str(minute) + 'm ' + str(second) + 's'
        else:
            day = int(time) // (60*60*24)
            hour = (int(time) - day * (60*60*24)) // (60*60)
            minute = (int(time) - day * (60*60*24) - hour * (60*60)) // 60
            second = int(time) - day * (60*60*24) - hour * (60*60) - minute * 60
            return str(day) + 'd ' + str(hour) + 'h ' + str(minute) + 'm ' + str(second) + 's'

            


    