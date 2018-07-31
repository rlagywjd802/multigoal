class videoScheduler(object):
    """
    Special class the only purpose of which is to trigger video recording
    """
    def __init__(self):
        self.record = False
        self._render_every_iterations = 10
        self._render_rollouts_num = 2

    def video_schedule(self, iter):
        return self.record

    @property
    def render_every_iterations(self):
        return self._render_every_iterations

    @render_every_iterations.setter
    def render_every_iterations(self, iter_num):
        self._render_every_iterations = iter_num

    @property
    def render_rollouts_num(self):
        return self._render_rollouts_num

    @render_rollouts_num.setter
    def render_rollouts_num(self, ro_num):
        self._render_rollouts_num = ro_num


video_scheduler = videoScheduler()