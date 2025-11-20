import torch as th

class PpoModel(th.nn.Module):
    def forward(self, ob) -> "pd, vpred, aux":
        raise NotImplementedError

    def act(self, ob, return_dist_params: bool = False) -> "action, dict(v_pred, logp)":
        pd, vpred, _ = self(ob)
        ac = pd.sample()
        logp = pd.log_prob(ac).sum(-1)
        info = {"vpred": vpred, "logp": logp}
        if return_dist_params:
            info["pd_mean"] = pd.mean
            info["pd_std"] = pd.stddev.log()
        return ac, info

    def v(self, ob):
        _pd, vpred, _ = self(ob)
        return vpred


class PhasicModel(PpoModel):
    def forward(self, ob) -> "pd, vpred, aux":
        raise NotImplementedError

    def compute_aux_loss(self, aux, mb):
        raise NotImplementedError

    def aux_keys(self) -> "list of keys needed in mb dict for compute_aux_loss":
        raise NotImplementedError

    def set_aux_phase(self, is_aux_phase: bool):
        "sometimes you want to modify the model, e.g. add a stop gradient"
        pass


