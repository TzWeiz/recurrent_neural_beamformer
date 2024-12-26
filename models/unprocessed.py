


class Unprocessed:
    def __init__(self, ref_mic_idx, **kwargs):

        super().__init__(**kwargs)
        self.ref_mic_idx = ref_mic_idx

    def forward(self, input):

        x = input["mixture"]

        output = {
            "est_target_multi_channel": x,
            "est_target": x[:, self.ref_mic_idx],
        }
        return output