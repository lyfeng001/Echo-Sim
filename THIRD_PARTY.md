# Third-party provenance

| Project | Use in this repository |
| --- | --- |
| [LIBERO-Recovery](https://github.com/liulin815/LIBERO-Recovery) | Base snapshot; recovery scene restoration and evaluation conventions |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `LIBERO/`: task environments, object models, textures and scene definitions |
| [StarVLA](https://github.com/starVLA/starVLA) | `starVLA/`: PI05/PI0 shared model, OpenPI action head, PaliGemma and image utilities |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | π0.5 architecture, normalization and public initialization |
| [Transformers](https://github.com/huggingface/transformers) | Vendored Gemma and SigLIP modules with OpenPI-compatible changes |

The base is LIBERO-Recovery revision
`7584d1baf3191e9a326c1d984e3fc7735a6cccbf`; see `UPSTREAM_SNAPSHOT.json`.
The PI05 utilities in `echo/models/pi05_utils.py` originate from that snapshot's
`examples/simBenchmarks/LIBERO/eval_files/openpi/pi_libero_common.py`.
`starVLA/model/image_tools.py` originates from `deployment/model_server/tools/`.
Their import paths were adjusted when unused examples and deployment code were
removed. The shared PI0 implementation remains because PI05 inherits from it.

Copyright and license notices in retained third-party files continue to apply.
Gemma and SigLIP carry Apache-2.0 notices; the corresponding license text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Dataset and model weight usage
terms are provided by their respective source repositories.

The upstream license texts for [LIBERO](LICENSES/LIBERO.txt) and
[StarVLA](LICENSES/starVLA.txt) are retained verbatim.
