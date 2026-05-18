import torch

class KVCache:
    def __init__(self, d_model, max_requests, max_request_len, device, dtype):
        self.d_model = d_model
        self.max_requests = max_requests
        self.max_requests_len = max_request_len

        self.uuid_to_slot = {}
        self.uuid_to_len= {}
        self.length_by_uuid = self.uuid_to_len
        self.free_slots = [i for i in range (max_requests)]

        self.K = torch.zeros((max_requests, max_request_len, d_model), device=device, dtype=dtype)
        self.V = torch.zeros((max_requests, max_request_len, d_model), device=device, dtype=dtype)

    def _get_index(self, uuid):
        if uuid in self.uuid_to_slot:
            return self.uuid_to_slot[uuid]
        elif self.free_slots == []:
            raise RuntimeError("Logical error in requests handling: Request was made for next slot but, KV cache is full")
        else:
            index = self.free_slots.pop()
            self.uuid_to_slot[uuid] = index
            self.uuid_to_len[uuid] = 0
            return index

    def _get_tensor_and_meta(self, uuid):
        index = self._get_index(uuid)
        return self.K[index], self.V[index], index, self.uuid_to_len[uuid]

    def _free_index(self, uuid):
        if uuid in self.uuid_to_slot:
            index = self.uuid_to_slot.pop(uuid)
            self.uuid_to_len.pop(uuid, None)
            self.free_slots += [index]

    def prefill(self, uuids, K, V, mask):
        for i, uuid in enumerate(uuids):
            seq_len = int(mask[i].sum().item())
            if seq_len > self.max_requests_len:
                raise RuntimeError(
                    f"Sequence length {seq_len} exceeds KV cache capacity {self.max_requests_len}"
                )
            K_cached, V_cached, index, _ = self._get_tensor_and_meta(uuid)
            K_cached[:seq_len] = K[i][:seq_len]
            V_cached[:seq_len] = V[i][:seq_len]
            self.uuid_to_len[uuid] = seq_len

    def append_and_fetch(self, uuids, K, V):
        if not isinstance(uuids, (list, tuple)):
            uuids = [uuids]
        K_ = []
        V_ = []
        q_positions = []
        lengths = []
        for i, uuid in enumerate(uuids):
            K_cached, V_cached, index, seq_len = self._get_tensor_and_meta(uuid)
            if seq_len == self.max_requests_len:
                raise RuntimeError("Logical error in request handling: Sequence to long, should not be possible")
            K_cached[seq_len:seq_len+1] = K[i]
            V_cached[seq_len:seq_len+1] = V[i]
            self.uuid_to_len[uuid] = seq_len + 1
            q_positions += [seq_len]
            lengths += [seq_len + 1]

        max_len = max(lengths)
        kv_mask = []
        k_positions = []
        for uuid, seq_len in zip(uuids, lengths):
            slot = self.uuid_to_slot[uuid]
            K_ += [self.K[slot, :max_len]]
            V_ += [self.V[slot, :max_len]]
            kv_mask += [torch.arange(max_len, device=K.device) < seq_len]
            k_positions += [torch.arange(max_len, device=K.device, dtype=torch.long)]

        return (
            torch.stack(K_),
            torch.stack(V_),
            torch.tensor(q_positions, device=K.device, dtype=torch.long).unsqueeze(1),
            torch.stack(k_positions),
            torch.stack(kv_mask),
        )

    def release(self, uuids):
        if not isinstance(uuids, (list, tuple)):
            uuids = [uuids]
        for uuid in uuids:
            self._free_index(uuid)
