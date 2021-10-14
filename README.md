# Breaking the Dilemma of Medical Image-to-image Translation

Supervised Pix2Pix and unsupervised Cycle-consistency are two modes that dominate the field of medical image-to-image translation. However, neither modes
are ideal. The Pix2Pix mode has excellent performance. But it requires paired
and well pixel-wise aligned images, which may not always be achievable due
to respiratory motion or anatomy change between times that paired images are
acquired. The Cycle-consistency mode is less stringent with training data and
works well on unpaired or misaligned images. But its performance may not be
optimal. In order to break the dilemma of the existing modes, we propose a new
unsupervised mode called RegGAN for medical image-to-image translation. It
is based on the theory of "loss-correction". In RegGAN, the misaligned target
images are considered as noisy labels and the generator is trained with an additional registration network to fit the misaligned noise distribution adaptively. The
goal is to search for the common optimal solution to both image-to-image translation and registration tasks. We incorporated RegGAN into a few state-of-the-art
image-to-image translation methods and demonstrated that RegGAN could be
easily combined with these methods to improve their performances. Such as a
simple CycleGAN in our mode surpasses latest NICEGAN even though using less
network parameters. Based on our results, RegGAN outperformed both Pix2Pix on
aligned data and Cycle-consistency on misaligned or unpaired data. RegGAN is
insensitive to noises which makes it a better choice for a wide range of scenarios,
especially for medical image-to-image translation tasks in which well pixel-wise
aligned data are not available

This paper has been accepted by [NeurIPS 2021](https://openreview.net/forum?id=C0GmZH2RnVR&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions)).
Get the full paper on [Arxiv](https://arxiv.org/pdf/2110.06465.pdf).




For more details or any questions, please feel easy to contact us by email ^\_^

