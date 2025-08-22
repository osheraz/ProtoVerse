# utils/wandb_farthestk_video_table.py
from typing import Optional, List
from dataclasses import dataclass
import wandb


@dataclass
class _VidRow:
    epoch: int
    reward: Optional[float]
    video: "wandb.sdk.data_types.video.Video"


class WandbFarthestKVideoTable:
    def __init__(
        self,
        max_keep: int = 12,
        table_key: str = "videos/rollouts",
        fps: int = 30,
    ):
        self.max_keep = int(max_keep)
        self.table_key = table_key
        self.fps = int(fps)
        self._kept: List[_VidRow] = []

    def submit(
        self,
        path: str,
        epoch: int,
        reward: Optional[float] = None,
        step: Optional[int] = None,
        run=None,
    ):
        run = run or wandb.run
        if run is None or not path:
            return

        vid = wandb.Video(path, fps=self.fps, caption=f"epoch {int(epoch)}")
        pool = self._kept + [
            _VidRow(int(epoch), None if reward is None else float(reward), vid)
        ]
        sel_idx = self._select_farthest_by_epoch(pool, k=self.max_keep)
        self._kept = [pool[i] for i in sorted(sel_idx, key=lambda i: pool[i].epoch)]

        table = wandb.Table(columns=["idx", "epoch", "reward", "video"])
        for i, row in enumerate(self._kept):
            table.add_data(i, row.epoch, row.reward, row.video)

        if step is None:
            run.log({self.table_key: table})
            run.log({"_debug/table_update": len(self._kept)})
        else:
            run.log({self.table_key: table}, step=int(step))
            run.log({"_debug/table_update": len(self._kept)}, step=int(step))

    @staticmethod
    def _select_farthest_by_epoch(items: List[_VidRow], k: int):
        n = len(items)
        if n <= k:
            return set(range(n))
        idxs = sorted(range(n), key=lambda i: items[i].epoch)
        if k == 1:
            return {idxs[-1]}
        selected = {idxs[0], idxs[-1]}

        def mindist(i: int) -> int:
            e = items[i].epoch
            return min(abs(e - items[j].epoch) for j in selected)

        while len(selected) < k:
            candidates = [i for i in idxs if i not in selected]
            best = max(candidates, key=mindist)
            selected.add(best)
        return selected
