import os
import nemo.collections.asr as nemo_asr
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Tuple
import coremltools as ct
import numpy as np

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

pretrained_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    model_name="ecapa_tdnn"
)

audio_signal_mel = torch.randn(1, 80, 16000)
audio_signal_len_mel = torch.tensor([audio_signal_mel.shape[1]])

exported_model_path = os.path.join(output_dir, "ECAPA-TDNN-NeMo.pt")
pretrained_model.export(
    exported_model_path, input_example=(audio_signal_mel, audio_signal_len_mel)
)


class ECAPATDNNWithMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super(ECAPATDNNWithMelSpectrogram, self).__init__()
        self.model = torch.load(exported_model_path, map_location=torch.device("cpu"))
        self.pretrained_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="ecapa_tdnn"
        )

    def forward(
        self, input_signal: torch.Tensor, input_signal_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_signal, processed_signal_len = self.pretrained_model.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )
        return self.model(
            processed_signal=processed_signal, processed_signal_len=processed_signal_len
        )


custom_model = ECAPATDNNWithMelSpectrogram()
custom_model.eval()

audio_signal = torch.randn(1, 16000 * 100)
audio_signal_len = torch.tensor([audio_signal.shape[1]])

traced_model = torch.jit.trace(
    custom_model, example_inputs=(audio_signal, audio_signal_len)
)

mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[
        ct.TensorType(
            name="inputSignal",
            shape=(
                1,
                ct.RangeDim(16000, 16000 * 100),
            ),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="inputSignalLength",
            shape=(ct.RangeDim(16000, 16000 * 100),),
            dtype=np.int64,
        ),
    ]
)

os.remove(exported_model_path)
exported_model_path = os.path.join(output_dir, "ECAPA-TDNN-NeMo.mlpackage")
mlmodel.save(exported_model_path)
