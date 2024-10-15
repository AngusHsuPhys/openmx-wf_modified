1. Utilize `wf.py` to generate OpenMX Atomate workflows. Currently, it's designed to create workflows using materials data from the Materials Project.

2. To effectively use `wf.py`, you need to set up your Python environment as per the `requirements.txt` file. It's important to note that the compute sites require customized versions of the [`atomate`](https://github.com/tsaie79/atomate/tree/openmx) and [`pymatgen`](https://github.com/tsaie79/pymatgen/tree/openmx) packages.

3. For deeph-e3 preprocessing, use the `tsaie79/deeph-e3-preprocess` image available on Docker Hub. For more details, refer to the [`deeph`](https://github.com/tsaie79/deepH) repository on GitHub.