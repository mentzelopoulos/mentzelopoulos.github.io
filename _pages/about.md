---
layout: about
title: About
permalink: /
subtitle: PhD Canidate, MIT | ament@mit.edu

profile:
  align: right
  image: prof_pic_alt.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p> Andreas Mentzelopoulos </p>
    <p> Rm 4-321 <p>
    <p> MIT, 55 Massachusetts Ave </p>
    <p> Cambridge, MA, USA </p>

news: true # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page
---

Hi! My name is Andreas and I am a PhD Candidate at the Department of Mechanical Engineering at [MIT](https://web.mit.edu/). My work focuses mainly on dynamical systems modeling using deep learning, time series forecasting, and generative modeling. I am advised by Professor [Michael S. Triantafyllou](https://meche.mit.edu/people/faculty/MISTETRI@MIT.EDU) at the [MIT Tow Tank](https://web.mit.edu/towtank) and Professor [Themis Sapsis](https://sandlab.mit.edu/?page_id=6) at the [Stochastic Analysis and Nonlinear Dynamics Lab](https://sandlab.mit.edu/).

Through my research work, I carve out machine learning solutions to engineering problems. Currently, I am developing a [digital twin](https://onepetro.org/OTCONF/proceedings-abstract/24OTC/4-24OTC/545009) for risers, which are long flexible underwater pipelines vibrating constantly under the excitation of stochastic hydrodynamic loads. Risers are constantly vibrating due to a complex flow structure interaction phenomenon known as [vortex-induced vibrations](https://en.wikipedia.org/wiki/Vortex-induced_vibration). The resulting dynamics are *nonlinear, nonstationary, and have memory*. I am currently leveraging [transformers](https://arxiv.org/abs/1706.03762) to model and continuously forecast the vibrations based on sparse observations (data) collected on the body *in real time*.

I am also developing generative modeling codes to synthesize instances of vortex-induced vibrations data (nonstationary 2D time series) using deep learning, since real data from experiments or the field are very expensive to obtain. I have developed solutions using [Generative Adversarial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network), especially [wGANs](https://arxiv.org/abs/1701.07875), [Variational Autoencoders (VAEs)](https://arxiv.org/abs/1312.6114), and [Denoising-Diffusion probabilistic models](https://arxiv.org/abs/2006.11239). I have been able to learn the diffusion process using both U-Net architectures and transformer architectures. I have shown that VAE data preserve the original training data's physical properties at least partially and that a forecasting model trained on synthetic VAE data can forecast real spatio-temporal vortex-induced vibrations reasonably accurately.

My work on generative modeling for vibrations has spiked my interest in using the same models to generate high-resolution seascapes (images) to raise awareness for the local ocean habitats in Massachusetts and the Gulf of Maine. I am running this effort, named LOBSTgER (Learning Oceanic Bioecological Systems Through Generative Representations), collaboratively with underwater photographer [Keith Ellenbogen](https://www.keithellenbogen.com/). Using Keith's **stunning** images as training data I have been able to train a [latent diffusion model](https://arxiv.org/abs/2112.10752) to generate 640x1024 images of Lion's Mane Jellyfish and Mola Mola. Check out LOBSTgER on my projects page for more info! We are currently open to collaborating with a computer vision expert on this if anyone is interested!

I was born and raised in [Athens, Greece](https://en.wikipedia.org/wiki/Athens) where I first cultivated a keen interest in engineering, developed a deep connection with the seas and the marine environments, and was first charmed by the beauty of music. My academic journey began in 2016 when I joined the [University of Michigan](https://umich.edu/) as an undergraduate student. By 2020, I graduated summa cum laude with two bachelor's degrees, one in [Mechanical Engineering](https://me.engin.umich.edu/) and the second in [Naval Architecture & Marine Engineering](https://name.engin.umich.edu/) with a minor in Mathematics. My undergraduate research was on marine renewable energy harvesting at the [MRELab](https://websites.umich.edu/~mrel/) under the mentorship of Professor [M.M. Bernitsas](https://name.engin.umich.edu/people/bernitsas-michael/). In 2020, I joined [MIT's MechE Department](https://meche.mit.edu/) where I have completed an SM (2022). Having spent over 7 years in academic programs, I have developed a strong background in mathematical modeling, numerical methods, statistics, algorithms, machine learning, optimization, and programming through coursework, research, and internships.

Besides academics, I am an avid musician and have been a violinist for the [MIT Symphony Orchestra](https://mta.mit.edu/music/performance/mit-symphony-orchestra) for the past four years. I enjoy riding [motorcycles](https://www.youtube.com/watch?v=YcSDnvT5VZ0) and am probably the only red motorcycle riding around Cambridge/Boston in the dead winter cold in search of breweries and concerts (check out @ath_bikelife_expat on Instagram).



<!--
Write your biography here. Tell the world about yourself. Link to your favorite [subreddit](http://reddit.com). You can put a picture in, too. The code is already in, just name your picture `prof_pic.jpg` and put it in the `img/` folder.

Put your address / P.O. box / other info right below your picture. You can also disable any of these elements by editing `profile` property of the YAML header of your `_pages/about.md`. Edit `_bibliography/papers.bib` and Jekyll will render your [publications page](/al-folio/publications/) automatically.

Link to your social media connections, too. This theme is set up to use [Font Awesome icons](https://fontawesome.com/) and [Academicons](https://jpswalsh.github.io/academicons/), like the ones below. Add your Facebook, Twitter, LinkedIn, Google Scholar, or just disable all of them.
-->
