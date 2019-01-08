Thon Rings
==========
If you look at the defocused image of a thin sample with phase contrast, then you will see Fresnel fringes around the edges. See, for example, this image from the :ref:`siemens_star` tutorial:

.. image:: images/siemens_fringes.png
   :width: 200

These fringes arise from the effective contrast transfer function, or optical transfer function, of the imaging system:

.. math::
    
    \begin{align}
       I(\mathbf{x}) &= \big| T(\mathbf{x}) \otimes \mathcal{F}^{-1}[\text{TF}](\mathbf{x}) \big|^2
    \end{align}

where I is the image on a detector, T the transmission function of an object and TF is the transfer function of the imaging system. 

For example, take free-space propagation:

.. math::
    
    \begin{align}
       I(\mathbf{x}) &= \big| T(\mathbf{x}) \otimes \mathcal{F}^{-1}[e^{-i\pi\lambda z_2\mathbf{q}^2}] \big|^2
    \end{align}

this roughly corresponds to the situation in the image above. Now if we take the Fourier transform of this image (see the image below) then we can often observe concentric rings modulating a complicated pattern. These are the Thon rings and they can be used to roughly estimate the TF, for example this might give us the propagation distance :math:`z_2` in the above example.

Thin weakly scattering object
-----------------------------
For a thin weakly scattering object we can approximate:

.. math::
    
    \begin{align}
    T(\mathbf{x})  &\approx e^{-i\frac{2\pi}{\lambda} \int dz n(\mathbf{x})} &&\text{projection approximation (thin sample)} \\
    &&& \text{with refractive index } n(\mathbf{x}) = \delta_\lambda(\mathbf{x}) -i\beta_\lambda(\mathbf{x}) \\
    &\approx e^{-\frac{2\pi}{\lambda} t(\mathbf{x}) (i\delta_\lambda + \beta_\lambda)} &&\text{single material of projected thickness } t(\mathbf{x}) \\
    &\approx 1 - \frac{2\pi}{\lambda} t(\mathbf{x}) (\beta_\lambda + i \delta_\lambda) &&\text{weakly scattering}
    \end{align}

Now we can approximate the Fourier transform of the image as:

.. math::
    
    \begin{align}
       \hat{I}(\mathbf{q}) &\approx \frac{2\pi}{\lambda}\hat{t}(q)\left[ 
       i\delta_\lambda \left( \text{TF}^*(-\mathbf{q}) - \text{TF}(\mathbf{q}) \right)
       -\beta_\lambda \left( \text{TF}^*(-\mathbf{q}) + \text{TF}(\mathbf{q}) \right)
       \right]  \quad \text{for } \mathbf{q} \neq 0
    \end{align}

Pure phase contrast image
^^^^^^^^^^^^^^^^^^^^^^^^^
Let us consider the case of free-space propagation, and a pure phase contrast image (:math:`\beta_\lambda = 0`):

.. math::
    
    \begin{align}
       \text{TF}(\mathbf{q}) &= e^{-i\pi\lambda z_2 \mathbf{q}^2} && \text{TF}^*(-\mathbf{q}) - \text{TF}(\mathbf{q}) 
       = 2i\sin(\pi\lambda z_2 \mathbf{q}^2) \\
       \hat{I}(\mathbf{q}) &= -\frac{4\pi}{\lambda} \delta_\lambda \hat{t}(\mathbf{q}) \sin(\pi\lambda z_2 \mathbf{q}^2)
       && \text{for } \mathbf{q} \neq 0
    \end{align}

.. image:: images/siemens_thon.png
   :width: 300

The above image was actually made from:

.. math::
    
    \begin{align}
    \text{image}(q) &= \sum_n \big|\mathcal{F}[M(r) e^{-r^2 / 2 \sigma^2} I_n(r) / W(r) ](q) \big|^2
    \end{align}

where :math:`n` is the image number, :math:`M(r)` is the mask, :math:`W(r)` is the whitefield, and the Gaussian is applied to avoid Fourier aliasing artefacts. The image was then taken to the power of 0.1 to enhance the contrast.

Projection images
-----------------
For projection imaging we have the approximation:

.. math::
    
    \begin{align}
    I^{z_1}_\phi(\mathbf{x}, z_2) &\approx \big|T(\mathbf{x}) \otimes e^{\frac{i \pi}{\lambda z^\text{eff}_x} \left( x - \frac{\lambda z^\text{eff}_x}{2\pi} \phi_{,x}(\mathbf{x})\right)^2
                                                                 \times  \frac{i \pi}{\lambda z^\text{eff}_y} \left( y - \frac{\lambda z^\text{eff}_y}{2\pi} \phi_{,y}(\mathbf{x})\right)^2} \big|^2
    \end{align}

Let make a Taylor series expansion of the phase up to second order in :math:`x` and :math:`y`:

.. math::
    
    \begin{align}
    \phi(\mathbf{x}) &\approx \phi_{00} + \phi_{10}x + \phi_{01}y + \frac{1}{2}\phi_{20}x^2 + \frac{1}{2}\phi_{02}y^2  \\
    \text{for } \phi_{nm} &= \frac{\partial^{n+m} \phi(\mathbf{x})}{\partial x^n \partial y^2}\biggr\rvert_{\mathbf{x}=0}
    \end{align}

where zeroth, first and second order terms correspond to piston, tilt and defocus aberrations. We have also set :math:`\phi_{11}=0` for simplicity. Now let's expand the exponential term:

.. math::
    
    \begin{align}
    \frac{i \pi}{\lambda z^\text{eff}_x} \left(x - \frac{\lambda z^\text{eff}_x}{2\pi} \phi_{,x}(\mathbf{x})\right)^2 
    &= \frac{i \pi}{\lambda z^\text{eff}_x} \left(x - \frac{\lambda z^\text{eff}_x}{2\pi} (\phi_{10} + \phi_{20}x) \right)^2 \\
    &= \frac{i \lambda z^\text{eff}_x}{4\pi} \left( \phi_{10} + (\phi_{20} -  \frac{2\pi}{\lambda z_x^\text{eff}})x\right)^2\\
    &= \frac{i \lambda z^\text{eff}_x}{4\pi} \left( \phi_{10} + \frac{2\pi}{\lambda z_2}x\right)^2 
     = \frac{i \pi z^\text{eff}_x}{\lambda z_2^2} \left( x + \frac{\lambda z_2}{2\pi}\phi_{10} \right)^2 \quad \text{where}\\
    z^\text{eff}_x = \frac{z_2}{1+\frac{\lambda z_2}{2\pi}\phi_{20}} \quad &\text{and} \quad 
    \phi_{20} -  \frac{2\pi}{\lambda z_x^\text{eff}} = \phi_{20} -  \frac{2\pi}{\lambda }\left(\frac{1}{z_2}+\frac{\lambda }{2\pi}\phi_{20}\right) = \frac{2\pi}{\lambda z_2} \\
    \end{align}

Now we can take the Fourier transform and evaluate the effective transfer function of the imaging system:

.. math::
    
    \begin{align}
    \text{TF}(\mathbf{q}) &= e^{-i\pi\lambda q'^2} 
                          e^{ i \lambda z_2 (\phi_{10} q_x + \phi_{01} q_y)} \quad \text{where}\\
    M_x &= \frac{z_2}{z_x^\text{eff}}, 
    \quad M_y = \frac{z_2}{z_y^\text{eff}} \quad \text{and} \quad 
    q'^2 = z^\text{eff}_x (M_x q_x)^2 + z^\text{eff}_y (M_y q_y)^2
    \end{align}

OK, so what do our Thon rings look like?

.. math::
    
    \begin{align}
       \text{TF}^*(-\mathbf{q}) - \text{TF}(\mathbf{q}) &= 2i\sin(\pi \lambda q'^2) e^{ i \lambda z_2 (\phi_{10} q_x + \phi_{01} q_y)}\\
       \text{TF}^*(-\mathbf{q}) + \text{TF}(\mathbf{q}) &= 2\cos(\pi \lambda q'^2) e^{ i \lambda z_2 (\phi_{10} q_x + \phi_{01} q_y)}\\
    \end{align}

which yeilds:

.. math::
    
    \begin{align}
       \big| \hat{I}(\mathbf{q})\big|^2 = \frac{4\pi}{\lambda}\big| \hat{t}(\mathbf{q})\big|^2 \left(\delta_\lambda  \sin(\pi \lambda q'^2) + 
       \beta_\lambda   \cos(\pi \lambda q'^2)
       \right)
    \end{align}


Fitting
-------
Let's ignore the physics for a moment and say that we have some array :math:`I_{nm}` given by:

.. math::
    
    \begin{align}
       I_{nm} &= \big| \hat{I}(\mathbf{q}_{nm})\big|^2 
    \end{align}

Now we filter the image, in order to flatten the contrast. Then we solve the problem:

.. math::
    
    \begin{align}
       I_{nm'} &= f_\sqrt{n^2 + m'^2} \quad \text{where} \quad m'= \text{scale_fs} \times m
    \end{align}

meaning that we are looking for the scaling factor along the fast scan axis of the array that makes the image most circular.
Now we fit a and b in the following profile:

.. math::
    
    \begin{align}
       f_n &= \sin(c n^2) + d\cos(c n^2)
    \end{align}

Now we return to the physics: given scale_fs, a and b we would like to determine:

.. math::
    
    \begin{align}
       &z_1, z_2, \delta z \quad \text{where}\\
       &\phi_{20} = \frac{2\pi}{\lambda (z_1 + \delta z)} \quad \text{and} \quad
       \phi_{02} = \frac{2\pi}{\lambda (z_1 - \delta z)}
    \end{align}

With the results in the above sections we have that:

.. math::
    
    \begin{align}
       \text{scale_fs} &= \left(\frac{N M_{fs} \Delta_{ss}}{M M_{ss} \Delta_{fs}}\right)^2 \frac{z_{fs}^{eff}}{z_{ss}^{eff}} && \\
         z_{ss}^{eff} &= \left( \frac{1}{z_2} + \frac{1}{z_1 + \delta z} \right)^{-1} &
         z_{fs}^{eff} &= \left( \frac{1}{z_2} + \frac{1}{z_1 - \delta z} \right)^{-1} \\
         z &= z_1 + z_2 &
         d &= \frac{\beta_\lambda}{\delta_\lambda} \\
         c &= \pi \lambda z_{ss}^{eff} \left(\frac{M_{ss}}{N\Delta_{ss}}\right)^2 
    \end{align}

So we have:

.. math::
    
    \begin{align}
        z_2^2 / z^\text{eff}_{ss} &= \frac{z_2(z+\delta z)}{z-z_2+\delta z} = \frac{(N \Delta_{ss})^2}{\pi \lambda} c = a\\ 
        z_2^2 / z^\text{eff}_{fs} &= \frac{z_2(z-\delta z)}{z-z_2-\delta z} = \frac{(M \Delta_{fs})^2}{\pi \lambda} c \times \text{scale_fs} = b
    \end{align}

This has the solution:

.. math::
    
    \begin{align}
        z_1      &= \frac{2z^2 - ab + \sqrt{a^2b^2 + a^2z^2 - 2abz^2 + b^2z^2}}{a + b + 2z} \\
        \delta z &= \frac{ab - \sqrt{a^2b^2 + a^2z^2 - 2abz^2 + b^2z^2}}{a - b}
    \end{align}


