.. _coord:

Coordinate system
=================
Define the centre pixel of the detector roi as x, y = (0, 0), for a sample translation of 0 and with no aberrations::
    
    x_map[(roi[1]-roi[0])//2] = 0
    y_map[(roi[3]-roi[2])//2] = 0

Basis vectors :math:`\mathbf{b}` give the mapping of sample shifts in x-y lab coordinates (:math:`\Delta \mathbf{x}`) to the slow scan and fast scan axes of the detector (:math:`\Delta \mathbf{x}_\text{D} = [\Delta x_{ss}, \Delta x_{fs}]`):

.. math::
    
    \begin{align}
        \Delta x_{ss} &= \frac{\mathbf{b}_{ss} \cdot \Delta \mathbf{x}_{xy}}{\text{x_pixel_size}}  \\
        \Delta x_{fs} &= \frac{\mathbf{b}_{fs} \cdot \Delta \mathbf{x}_{xy}}{\text{y_pixel_size}} 
    \end{align}
