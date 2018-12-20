Thon Rings
==========
If you look at the defocused image of a thin sample with phase contrast, then you will see Fresnel fringes around the edges. See, for example, this image from the :ref:`siemens_star` tutorial:

.. image:: images/siemens_fringes.png
   :width: 200

Now these fringes will become broader the closer the object is to the focus, and more fine the further away it is. So they are a useful means to determine this information. Of course this method relies on our ability to actually see the fringes, if the coherence of the beam is too low, or the sample wobbles during the exposure, then these fringes will be washed out. 

Let's assume that we have:

1. A thin weakly scattering object: :math:`T(r) = |T(r)| e^{i\phi(r)} \approx 1 + i\phi(r)`, and
2. that plane wave light of wavelength :math:`\lambda` passes through it,
3. a pixilated detector imaging the intensity a distance z from the sample.

Let us now see what the defocused image looks like, ignoring terms of order :math:`\phi^2`:

.. math::
    
    \begin{align}
    I(r) &= \big| T(r) \otimes e^{i\pi \frac{r^2}{\lambda z}} \big|^2 \\ 
                &\approx 1 - 2 \Im\left\{\phi(r) \otimes e^{i\pi \frac{r^2}{\lambda z}}\right\} 
    \end{align}
    
Now take the Fourier transform of the defocused image to see the rings:

.. math::
    
    \begin{align}
    \mathcal{F}[I](q) &\approx \delta(q) + 2 \sin(\pi \lambda z q^2)\hat{\phi}(q)
    \end{align}

.. image:: images/siemens_thon.png
   :width: 300

The above image was actually made from:

.. math::
    
    \begin{align}
    \text{image}(q) &= \sum_n \big|\mathcal{F}[M(r) e^{-r^2 / 2 \sigma^2} I_n(r) / W(r) ](q) \big|^2
    \end{align}

where :math:`n` is the image number, :math:`M(r)` is the mask, :math:`W(r)` is the whitefield, and the Gaussian is applied to avoid Fourier aliasing artefacts. The image was then taken to the power of 0.1 to enhance the contrast.
