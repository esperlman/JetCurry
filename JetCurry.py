# Copyright 2017. Katie Kosak. Eric Perlman
#    This file is part of JetCurry.
#    JetCurry is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math
from pylab import *
from matplotlib import *
from scipy.interpolate import spline
import emcee
from scipy.optimize import fmin_l_bfgs_b
from multiprocessing import Process
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

np.seterr(all='ignore')

def imagesqrt(image, scale_min, scale_max):
    '''
    Algorithm Courtesy of Min-Su Shin (msshin @ umich.edu)
    Modified by Katie Kosak 07/02/2015 to fit the needsof Hubble Data for
    Dr. Perlman and Dr. Avachat.

    Arguments:
        image : numpy array
            FITS image
        scale_min : float
            Minimum scale
        scale_max : float
            Maximum scale

    Returns:
        imageData : numpy array
            Square root scale of image

    '''
    imageData = np.array(image, copy=True)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    imageData = np.sqrt(imageData)
    imageData = imageData / math.sqrt(scale_max - scale_min)
    return imageData


def Find_MaxFlux(file1, Upstream_Bounds, Downstream_Bounds, number_of_points):
    '''
    Calculates the max along each column

    Arguments:
        file1 : numpy array
            Sqrt image of FITS file
        Upstream_Bounds : numpy array
            Start position of jet
        Downstream_Bounds : numpy array
            End position of jet
        number_of_points : numpy integer
            Number of pixels along x-axis of the start and end positions of jet

    Returns:
        intensity_xpos : numpy array
            Max intensity at point x
        intensity_ypos : numpy array
            Max intensity at point y
        x_smooth : numpy array
            Returns number_of_points evenly spaced samples
                calculated over the start/stop interval
        y_smooth : numpy array
            Interpolate a curve using spline fit
        intensity_max : numpy array
            Max intensity of each column
    '''
    
    height, width = file1.shape
    intensity_max = np.array([])
    intensity_ypos = np.array([])
    intensity_xpos = np.array([])
    temp = np.zeros(shape=(height, 1))
    for k in range(int(Upstream_Bounds[0]), int(Downstream_Bounds[0]) - 1):
        for j in range(0, height - 1):
            pixel = file1[j, k]
            temp[j] = pixel
        if np.sum(temp) != 0.0:
            pixel_max = np.nanmax(temp)
            position = [i for i, j in enumerate(temp) if j == pixel_max]
            intensity_max = np.append(intensity_max, pixel_max)
            intensity_ypos = np.append(intensity_ypos, position[0])
            intensity_xpos = np.append(intensity_xpos, k)
        else: continue         
    
    #Calculates linear spline fit coordinates:
    function = interp1d(intensity_xpos, intensity_ypos, kind = 'linear')
    x_smooth = np.linspace(
        intensity_xpos[0],
        intensity_xpos[-1],
        num = number_of_points)
    y_smooth = function(x_smooth)
    return intensity_xpos, intensity_ypos, x_smooth, y_smooth, intensity_max


def Calculate_s_and_eta(x_smooth, y_smooth, core_points, output_directory, filename):
    '''
    Calculates S and ETA values

    Arguments:
        x_smooth : numpy array
            Evenly spaced samples
        y_smooth : numpy array
            Interpolate a curve using spline fit
        core_points : numpy array
            Start position of jet
        output_directory : string
            Path of location to save data products
        filename : string
            Root name of file to be saved

    Returns:
        s: list
        eta: list
    '''
    s = []
    eta = []

    for i in range(len(x_smooth)):
        x = x_smooth[i] - float(core_points[1])  # default core_points[1]
        y = y_smooth[i] - float(core_points[0])  # default core_points[0]
        s_value = (x**2 + y**2)**(0.5)
        eta_value = math.atan(y / x)
        s.append(s_value)
        eta.append(eta_value)

    with open(output_directory + filename + '_parameters.txt', 'w') as file:
        for x in range(len(x_smooth)):
            file.write(str(s[x]) + '\t' + str(eta[x]) + '\t' +
                       str(x_smooth[x]) + '\t' + str(y_smooth[x]) + '\n')
    return s, eta


def Run_MCMC1(s, eta, theta, ind, output_directory, filename):
    '''
    '''
    ndim, nwalkers, nsteps = 5, 1024, 50
    initial_alpha = np.arange(0, 2.0, 0.5)
    initial_beta = np.arange(0, 2.0, 0.5)
    initial_phi = np.arange(0, 2.0, 0.5)
    initial_xi = np.arange(0, 2.0, 0.5)
    initial_d = np.arange(np.floor(s), np.floor(s) + 4 * 20.25, 20.25)
    pos = []
    for i in range(len(initial_alpha)):
        for j in range(len(initial_beta)):
            for k in range(len(initial_phi)):
                for l in range(len(initial_xi)):
                    for m in range(len(initial_d)):
                        init = []
                        init.append(initial_alpha[i])
                        init.append(initial_beta[j])
                        init.append(initial_phi[k])
                        init.append(initial_xi[l])
                        init.append(initial_d[m])
                        pos.append(init)

    def lnlike(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        model = (((s * np.sin(eta)) / np.sin(phi))**2 + (s * np.cos(eta) * (np.sin(theta) * np.cos(alpha) + np.sin(alpha) * np.cos(theta)) / np.cos(alpha))**2 - d**2)**2 + (((np.sin(beta) * np.cos(alpha)) / (np.sin(alpha) * np.cos(beta)))**2 - (np.cos(eta)**2))**2 + (d * np.cos(xi) * np.cos(theta) - ((s * np.cos(eta) * np.sin(alpha)) / np.cos(alpha)) - d * np.sin(xi) * np.cos(phi) * np.sin(theta))**2 + ((np.sin(eta) / np.cos(eta)) - ((np.sin(xi) * np.sin(phi)) / (np.cos(xi) * np.sin(theta) + np.sin(xi) * np.cos(phi) * np.cos(theta))))**2 + (s - d * np.cos(beta))**2
        return -np.log(np.abs(model) + 1)

    t = 1.5708 - theta

    def lnprior(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        if 0 < alpha < 1.57 and 0 < beta < 1.57 and 0 < phi < 3.14 and 0 < xi < t and np.floor(s) < d < np.floor(s) + 3 * 20.25:
            return 0.0
        return np.inf

    def lnprob(v):
        '''
        '''
        lp = lnprior(v)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(v)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    probs = sampler.flatlnprobability
    A = np.array(probs)
    maximum_indices = np.where(A == max(probs))
    pvectors = []
    pvectors = samples[maximum_indices]
    r = pvectors[0]

    with open(output_directory + filename + '_MCMC1.txt', 'a') as file:
        file.write(str(r[0]) +
                   '\t' +
                   str(r[1]) +
                   '\t' +
                   str(r[2]) +
                   '\t' +
                   str(r[3]) +
                   '\t' +
                   str(r[4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def MCMC1_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    for i in range(0, int(len(s) - 3.0), 4):
        r = Process(target=Run_MCMC1, args=(s[i], eta[i], theta, i, output_directory, filename))
        r.start()
        r.join()

        r1 = Process(target=Run_MCMC1, args=(
            s[i + 1], eta[i + 1], theta, i + 1, output_directory, filename))
        r1.start()
        r1.join()

        r2 = Process(target=Run_MCMC1, args=(
            s[i + 2], eta[i + 2], theta, i + 2, output_directory, filename))
        r2.start()
        r2.join()

        r3 = Process(target=Run_MCMC1, args=(
            s[i + 3], eta[i + 3], theta, i + 3, output_directory, filename))
        r3.start()
        r3.join()

    return


def RunMCMC2(s, eta, d0, theta, a, b, c, e, ind, output_directory, filename):

    initial_alpha = np.arange(a, a + 0.2, 0.05)
    initial_beta = np.arange(b, b + 0.2, 0.05)
    initial_phi = np.arange(c, c + 0.2, 0.05)
    initial_xi = np.arange(e, e + 0.2, 0.05)
    initial_d = np.arange(np.floor(d0), np.floor(d0) + 2.0, 0.5)

    pos = []
    for i in range(len(initial_alpha)):
        for j in range(len(initial_beta)):
            for k in range(len(initial_phi)):
                for l in range(len(initial_xi)):
                    for m in range(len(initial_d)):
                        init = []
                        init.append(initial_alpha[i])
                        init.append(initial_beta[j])
                        init.append(initial_phi[k])
                        init.append(initial_xi[l])
                        init.append(initial_d[m])
                        pos.append(init)
   
    ndim, nwalkers, nsteps = 5, len(pos), 50
    def lnlike(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        model = (((s * np.sin(eta)) / np.sin(phi))**2 + (s * np.cos(eta) * (np.sin(theta) * np.cos(alpha) + np.sin(alpha) * np.cos(theta)) / np.cos(alpha))**2 - d**2)**2 + (((np.sin(beta) * np.cos(alpha)) / (np.sin(alpha) * np.cos(beta)))**2 - (np.cos(eta)**2))**2 + (d * np.cos(xi) * np.cos(theta) - ((s * np.cos(eta) * np.sin(alpha)) / np.cos(alpha)) - d * np.sin(xi) * np.cos(phi) * np.sin(theta))**2 + ((np.sin(eta) / np.cos(eta)) - ((np.sin(xi) * np.sin(phi)) / (np.cos(xi) * np.sin(theta) + np.sin(xi) * np.cos(phi) * np.cos(theta))))**2 + (s - d * np.cos(beta))**2
        return -np.log(np.abs(model) + 1)

    t = 1.5708 - theta

    def lnprior(v):
        '''
        '''
        alpha, beta, phi, xi, d = v
        if a < alpha < (
                a +
                0.2) and b < beta < (
                b +
                0.2) and c < phi < (
                c +
                0.2) and e < xi < (
                    e +
                    0.2) and np.floor(d0) < d < (
                        np.floor(d0) +
                        3 *
                20.25):
            return 0.0
        return np.inf

    def lnprob(v):
        '''
        '''
        lp = lnprior(v)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(v)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    probs = sampler.flatlnprobability
    A = np.array(probs)
    maximum_indices = np.where(A == max(probs))
    pvectors = []
    pvectors = samples[maximum_indices]
    h = pvectors[0]

    with open(output_directory + filename + '_MCMC2.txt', 'a') as file:
        file.write(str(h[0]) +
                   '\t' +
                   str(h[1]) +
                   '\t' +
                   str(h[2]) +
                   '\t' +
                   str(h[3]) +
                   '\t' +
                   str(h[4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def MCMC2_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_first = []
    beta_first = []
    phi_first = []
    xi_first = []
    d_first = []
    for line in open(output_directory + filename + '_MCMC1.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        # if line.startswith('\n'):
        #    continue
        fields = line.split()
        alpha_first.append(fields[0])
        beta_first.append(fields[1])
        phi_first.append(fields[2])
        xi_first.append(fields[3])
        d_first.append(fields[4])
    alpha_first = [float(i) for i in alpha_first]
    beta_first = [float(i) for i in beta_first]
    phi_first = [float(i) for i in phi_first]
    xi_first = [float(i) for i in xi_first]
    d_first = [float(i) for i in d_first]
    for j in range(0, len(alpha_first), 4):
        alpha0 = float(alpha_first[j])
        beta0 = float(beta_first[j])
        phi0 = float(phi_first[j])
        xi0 = float(xi_first[j])
        d0 = float(d_first[j])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        h = Process(
            target=RunMCMC2,
            args=(
                s[j],
                eta[j],
                d0,
                theta,
                a,
                b,
                c,
                e,
                j, output_directory, filename))
        h.start()
        h.join()
        alpha1 = float(alpha_first[j + 1])
        beta1 = float(beta_first[j + 1])
        phi1 = float(phi_first[j + 1])
        xi1 = float(xi_first[j + 1])
        d1 = float(d_first[j + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        h1 = Process(target=RunMCMC2, args=(
            s[j + 1], eta[j + 1], d1, theta, a1, b1, c1, e1, j + 1, output_directory, filename))
        h1.start()
        h1.join()
        alpha2 = float(alpha_first[j + 2])
        beta2 = float(beta_first[j + 2])
        phi2 = float(phi_first[j + 2])
        xi2 = float(xi_first[j + 2])
        d2 = float(d_first[j + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        h2 = Process(target=RunMCMC2, args=(
            s[j + 2], eta[j + 2], d2, theta, a2, b2, c2, e2, j + 2, output_directory, filename))
        h2.start()
        h2.join()
        alpha3 = float(alpha_first[j + 3])
        beta3 = float(beta_first[j + 3])
        phi3 = float(phi_first[j + 3])
        xi3 = float(xi_first[j + 3])
        d3 = float(d_first[j + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        h3 = Process(target=RunMCMC2, args=(
            s[j + 3], eta[j + 3], d3, theta, a3, b3, c3, e3, j + 3, output_directory, filename))
        h3.start()
        h3.join()
    return

def Annealing1(eta, s, theta, alpha0, beta0, phi0, xi0, d0, a, b, c, e, ind, output_directory, filename):
    '''
    '''

    def f(x):
        '''
        '''
        alpha, beta, phi, xi, d = x
        return ((((s *
                   np.sin(eta)) /
                  np.sin(phi))**2 +
                 (s *
                  np.cos(eta) *
                  (np.sin(theta) *
                   np.cos(alpha) +
                   np.sin(alpha) *
                   np.cos(theta)) /
                  np.cos(alpha))**2 -
                 d**2)**2 +
                (((np.sin(beta) *
                   np.cos(alpha)) /
                  (np.sin(alpha) *
                   np.cos(beta)))**2 -
                 (np.cos(eta)**2))**2 +
                (d *
                 np.cos(xi) *
                 np.cos(theta) -
                 ((s *
                   np.cos(eta) *
                   np.sin(alpha)) /
                  np.cos(alpha)) -
                 d *
                 np.sin(xi) *
                 np.cos(phi) *
                 np.sin(theta))**2 +
                ((np.sin(eta) /
                  np.cos(eta)) -
                 ((np.sin(xi) *
                   np.sin(phi)) /
                    (np.cos(xi) *
                     np.sin(theta) +
                     np.sin(xi) *
                     np.cos(phi) *
                     np.cos(theta))))**2 +
                (s -
                 d *
                 np.cos(beta))**2)
    x0 = np.array([float(alpha0), float(beta0),
                   float(phi0), float(xi0), float(d0)])
    xmin = [a, b, c, e, float(floor(d0))]
    xmax = [float(a + 0.2), float(b + 0.2), float(c + 0.2),
            float(e + 0.2), float(floor(d0) + 2.0)]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]
    res = fmin_l_bfgs_b(
        f,
        x0,
        fprime=None,
        args=(),
        approx_grad=True,
        bounds=bounds,
        m=10,
        factr=10000000.0,
        pgtol=1e-05,
        epsilon=1e-08,
        iprint=-1,
        maxfun=150,
        maxiter=150,
        disp=None,
        callback=None)
    q = res

    with open(output_directory + filename + '_ANNE1.txt', 'a') as file:
        file.write(str(q[0][0]) +
                   '\t' +
                   str(q[0][1]) +
                   '\t' +
                   str(q[0][2]) +
                   '\t' +
                   str(q[0][3]) +
                   '\t' +
                   str(q[0][4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return


def Annealing1_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_MCMC2 = []
    beta_MCMC2 = []
    phi_MCMC2 = []
    xi_MCMC2 = []
    d_MCMC2 = []
    for line in open(output_directory + filename + '_MCMC2.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        object_alpha = fields[0]
        alpha_MCMC2.append(object_alpha)
        object_beta = fields[1]
        beta_MCMC2.append(object_beta)
        object_phi = fields[2]
        phi_MCMC2.append(object_phi)
        object_xi = fields[3]
        xi_MCMC2.append(object_xi)
        object_d = fields[4]
        d_MCMC2.append(object_d)
    for k in range(0, len(alpha_MCMC2), 4):
        alpha0 = float(alpha_MCMC2[k])
        beta0 = float(beta_MCMC2[k])
        phi0 = float(phi_MCMC2[k])
        xi0 = float(xi_MCMC2[k])
        d0 = float(d_MCMC2[k])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        q = Process(
            target=Annealing1,
            args=(
                eta[k],
                s[k],
                theta,
                alpha0,
                beta0,
                phi0,
                xi0,
                d0,
                a,
                b,
                c,
                e,
                k, output_directory, filename))
        q.start()
        q.join()
        alpha1 = float(alpha_MCMC2[k + 1])
        beta1 = float(beta_MCMC2[k + 1])
        phi1 = float(phi_MCMC2[k + 1])
        xi1 = float(xi_MCMC2[k + 1])
        d1 = float(d_MCMC2[k + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        q1 = Process(target=Annealing1,
                     args=(eta[k + 1],
                           s[k + 1],
                           theta,
                           alpha1,
                           beta1,
                           phi1,
                           xi1,
                           d1,
                           a1,
                           b1,
                           c1,
                           e1,
                           k + 1, output_directory, filename))
        q1.start()
        q1.join()
        alpha2 = float(alpha_MCMC2[k + 2])
        beta2 = float(beta_MCMC2[k + 2])
        phi2 = float(phi_MCMC2[k + 2])
        xi2 = float(xi_MCMC2[k + 2])
        d2 = float(d_MCMC2[k + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        q2 = Process(target=Annealing1,
                     args=(eta[k + 2],
                           s[k + 2],
                           theta,
                           alpha2,
                           beta2,
                           phi2,
                           xi2,
                           d2,
                           a2,
                           b2,
                           c2,
                           e2,
                           k + 2, output_directory, filename))
        q2.start()
        q2.join()
        alpha3 = float(alpha_MCMC2[k + 3])
        beta3 = float(beta_MCMC2[k + 3])
        phi3 = float(phi_MCMC2[k + 3])
        xi3 = float(xi_MCMC2[k + 3])
        d3 = float(d_MCMC2[k + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        q3 = Process(target=Annealing1,
                     args=(eta[k + 3],
                           s[k + 3],
                           theta,
                           alpha3,
                           beta3,
                           phi3,
                           xi3,
                           d3,
                           a3,
                           b3,
                           c3,
                           e3,
                           k + 3, output_directory, filename))
        q3.start()
        q3.join()
    return

def Annealing2(eta, s, theta, alpha0, beta0, phi0, xi0, d0, a, b, c, e, ind, output_directory, filename):
    '''
    '''

    def f(x):
        '''
        '''
        alpha, beta, phi, xi, d = x
        return ((((s *
                   np.sin(eta)) /
                  np.sin(phi))**2 +
                 (s *
                  np.cos(eta) *
                  (np.sin(theta) *
                   np.cos(alpha) +
                   np.sin(alpha) *
                   np.cos(theta)) /
                  np.cos(alpha))**2 -
                 d**2)**2 +
                (((np.sin(beta) *
                   np.cos(alpha)) /
                  (np.sin(alpha) *
                   np.cos(beta)))**2 -
                 (np.cos(eta)**2))**2 +
                (d *
                 np.cos(xi) *
                 np.cos(theta) -
                 ((s *
                   np.cos(eta) *
                   np.sin(alpha)) /
                  np.cos(alpha)) -
                 d *
                 np.sin(xi) *
                 np.cos(phi) *
                 np.sin(theta))**2 +
                ((np.sin(eta) /
                  np.cos(eta)) -
                 ((np.sin(xi) *
                   np.sin(phi)) /
                    (np.cos(xi) *
                     np.sin(theta) +
                     np.sin(xi) *
                     np.cos(phi) *
                     np.cos(theta))))**2 +
                (s -
                 d *
                 np.cos(beta))**2)
    x0 = np.array([float(alpha0), float(beta0),
                   float(phi0), float(xi0), float(d0)])
    xmin = [a, b, c, e, float(floor(d0))]
    xmax = [float(a + 0.1), float(b + 0.1), float(c + 0.1),
            float(e + 0.1), float(floor(d0) + 1.0)]
    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    res = fmin_l_bfgs_b(
        f,
        x0,
        fprime=None,
        args=(),
        approx_grad=True,
        bounds=bounds,
        m=10,
        factr=10000000.0,
        pgtol=1e-05,
        epsilon=1e-08,
        iprint=-1,
        maxfun=150,
        maxiter=150,
        disp=None,
        callback=None)
    q = res

    with open(output_directory + filename + '_ANNE2.txt', 'a') as file:
        file.write(str(q[0][0]) +
                   '\t' +
                   str(q[0][1]) +
                   '\t' +
                   str(q[0][2]) +
                   '\t' +
                   str(q[0][3]) +
                   '\t' +
                   str(q[0][4]) +
                   '\t' +
                   str(ind) +
                   '\n')
    return

def Annealing2_Parallel(s, eta, theta, output_directory, filename):
    '''
    '''
    alpha_ANNE1 = []
    beta_ANNE1 = []
    phi_ANNE1 = []
    xi_ANNE1 = []
    d_ANNE1 = []

    for line in open(output_directory + filename + '_ANNE1.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        object_alpha = fields[0]
        alpha_ANNE1.append(object_alpha)
        object_beta = fields[1]
        beta_ANNE1.append(object_beta)
        object_phi = fields[2]
        phi_ANNE1.append(object_phi)
        object_xi = fields[3]
        xi_ANNE1.append(object_xi)
        object_d = fields[4]
        d_ANNE1.append(object_d)
    for k in range(0, len(alpha_ANNE1), 4):
        alpha0 = float(alpha_ANNE1[k])
        beta0 = float(beta_ANNE1[k])
        phi0 = float(phi_ANNE1[k])
        xi0 = float(xi_ANNE1[k])
        d0 = float(d_ANNE1[k])
        a = float(0.1 * (floor(10 * alpha0)))
        b = float(0.1 * floor(10 * beta0))
        c = float(0.1 * floor(10 * phi0))
        e = float(0.1 * floor(10 * xi0))
        q = Process(
            target=Annealing2,
            args=(
                eta[k],
                s[k],
                theta,
                alpha0,
                beta0,
                phi0,
                xi0,
                d0,
                a,
                b,
                c,
                e,
                k, output_directory, filename))
        q.start()
        q.join()
        alpha1 = float(alpha_ANNE1[k + 1])
        beta1 = float(beta_ANNE1[k + 1])
        phi1 = float(phi_ANNE1[k + 1])
        xi1 = float(xi_ANNE1[k + 1])
        d1 = float(d_ANNE1[k + 1])
        a1 = float(0.1 * (floor(10 * alpha1)))
        b1 = float(0.1 * floor(10 * beta1))
        c1 = float(0.1 * floor(10 * phi1))
        e1 = float(0.1 * floor(10 * xi1))
        q1 = Process(target=Annealing2,
                     args=(eta[k + 1],
                           s[k + 1],
                           theta,
                           alpha1,
                           beta1,
                           phi1,
                           xi1,
                           d1,
                           a1,
                           b1,
                           c1,
                           e1,
                           k + 1, output_directory, filename))
        q1.start()
        q1.join()
        alpha2 = float(alpha_ANNE1[k + 2])
        beta2 = float(beta_ANNE1[k + 2])
        phi2 = float(phi_ANNE1[k + 2])
        xi2 = float(xi_ANNE1[k + 2])
        d2 = float(d_ANNE1[k + 2])
        a2 = float(0.1 * (floor(10 * alpha2)))
        b2 = float(0.1 * floor(10 * beta2))
        c2 = float(0.1 * floor(10 * phi2))
        e2 = float(0.1 * floor(10 * xi2))
        q2 = Process(target=Annealing2,
                     args=(eta[k + 2],
                           s[k + 2],
                           theta,
                           alpha2,
                           beta2,
                           phi2,
                           xi2,
                           d2,
                           a2,
                           b2,
                           c2,
                           e2,
                           k + 2, output_directory, filename))
        q2.start()
        q2.join()
        alpha3 = float(alpha_ANNE1[k + 3])
        beta3 = float(beta_ANNE1[k + 3])
        phi3 = float(phi_ANNE1[k + 3])
        xi3 = float(xi_ANNE1[k + 3])
        d3 = float(d_ANNE1[k + 3])
        a3 = float(0.1 * (floor(10 * alpha3)))
        b3 = float(0.1 * floor(10 * beta3))
        c3 = float(0.1 * floor(10 * phi3))
        e3 = float(0.1 * floor(10 * xi3))
        q3 = Process(target=Annealing2,
                     args=(eta[k + 3],
                           s[k + 3],
                           theta,
                           alpha3,
                           beta3,
                           phi3,
                           xi3,
                           d3,
                           a3,
                           b3,
                           c3,
                           e3,
                           k + 3, output_directory, filename))
        q3.start()
        q3.join()
    return


def Convert_Results_Cartesian(s, eta, theta, output_directory, filename):
    '''
    '''
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    alpha = []
    beta = []
    phi = []
    xi = []
    d = []
    for line in open(output_directory + filename + '_ANNE2.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        # if line.startswith('\n'):
        #   continue
        fields = line.split()
        alpha.append(fields[0])
        beta.append(fields[1])
        phi.append(fields[2])
        xi.append(fields[3])
        d.append(fields[4])
    for line in open(output_directory + filename + '_parameters.txt', 'r').readlines():
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue
        fields = line.split()
        eta.append(fields[1])
    # Convert to Float data type
    d = [float(i) for i in d]
    alpha = [float(i) for i in alpha]
    beta = [float(i) for i in beta]
    xi = [float(i) for i in xi]
    eta = [float(i) for i in eta]
    phi = [float(i) for i in phi]

    for i in range(len(alpha)):

        x_value = d[i] * np.cos(eta[i]) * np.cos(beta[i]) + 15
        z_value = d[i] * np.cos(beta[i]) * np.cos(eta[i]) * np.tan(alpha[i])
        z_coordinates.append(z_value)

        y_value = d[i] * np.cos(beta[i]) * np.sin(eta[i])

        x_coordinates.append(x_value)
        y_coordinates.append(y_value + 13)

    with open(output_directory + filename + '_Cartesian_Coordinates.txt', 'w') as file:
    # Write Results to File
        for i in range(len(x_coordinates)):
            file.write(str(x_coordinates[i]) +
                       '\t' +
                       str(y_coordinates[i]) +
                       '\t' +
                       str(z_coordinates[i]) +
                       '\n')
    return x_coordinates, y_coordinates, z_coordinates


def read_Cartesian_Coordinates_File(cart_coords_file):
    '''
    This function reads the "_Cartesian_Coordinates.txt" 
    and returns its contents as three separate numpy arrays: x_mp,y_mp,z_mp
    
    Input Arguments:
    cart_coords_files = Name of the jet along with the file path 
                        written as an array of strings, 
                        where each string is the name of the jet.
                        
    Output Arguments:
    NOTE: "_mp" signifies that these are the most probable solutions 
          for (x,y,z) coordinates
    x_mp = Array of x coordinates from _Cartesian_Coordinatees.txt file
    y_mp = Array of y coordinates from _Cartesian_Coordinatees.txt file
    z_mp = Array of z coordinates from _Cartesian_Coordinatees.txt file
    ''' 
    
    #To initialize the arrays that will save
    #the most probable solutions for Cartesian coordiantes (x,y,z):
    x_mp=[]
    y_mp=[]
    z_mp=[]

    for line in open(cart_coords_file).readlines(): 
        if line.startswith('#'):
            continue
        if line.startswith('\n'):
            continue

        fields = line.split() 
        x_mp.append(fields[0])
        y_mp.append(fields[1]) 
        z_mp.append(fields[2])

    #To convert the appended lists to numpy arrays:
    x_mp=[float(i) for i in x_mp]
    y_mp=[float(i) for i in y_mp]
    z_mp=[float(i) for i in z_mp]
    
   
    #To remove outliers:
    #To remove coordinates that have negative y values:
    #To find the negative values in y_mp vector
    #and store it is a vector:
    neg_values = np.array(y_mp)[np.array(y_mp) < 0]
    for neg_value in neg_values:
        #To find the index corresponding to the negative value:
        i = y_mp.index(neg_value)
        #To remove the negative y values
        #and corresponding x and z values:
        x_mp.remove(x_mp[i])
        y_mp.remove(y_mp[i])
        z_mp.remove(z_mp[i])

    #To correct for the origin:
    x_mp = np.array(x_mp)
    y_mp = np.array(y_mp)
    #print(np.array(x_mp),np.array(y_mp),np.array(z_mp))
   
    return (np.array(x_mp),np.array(y_mp),np.array(z_mp))

def PCA(x,y,z):
    '''
    This function implements Principal Component Analysis 
    on a set of 3-D Cartesian coordiantes. 
    INPUT ARGUMENTS:
    x = An array of x Cartesian coordinates
    y = An array of y Cartesian coordinates
    z = An array of z Cartesian coordinates
    OUTPUT ARGUMENTS:
    coords_cal = A matrix of calculated PCA coordinates. 
    PCA_line_coords = A matrix of calculated PCA coordinates 
                      that lie on a straight line.These coordiantes lie 
                      on the line that represents the preferred direction.
    NOTE: 
    PCA_line_coords[:,0] represents the x PCA coordinate that lies on the line
    PCA_line_coords[:,1] represents the y PCA coordinate that lies on the line
    PCA_line_coords[:,2] represents the z PCA coordinate that lies on the line
    t = The scalar paramter to define the point on the preferred line.
    '''
    
    #To create an (3xn) matrix of these coordinates:
    coord_mat = np.array([x,y,z])

    #To calculate the mean of each column and save it as an (1x3) row matrix:
    x_col_mean = np.mean(coord_mat[0])
    y_col_mean = np.mean(coord_mat[1])
    z_col_mean = np.mean(coord_mat[2])

    mean_cols = np.array([x_col_mean,y_col_mean,z_col_mean])

    #To normalize the coord_mat matrix and save the result 
    #as an (number of datapoints x number of dimensions) matrix:
    #Dimensions:
    n = 3
    coord_norm = []
    for i in range(0,3,1):
        coord_norm.append(coord_mat[i] - mean_cols[i])
    coord_mat_norm = np.array(coord_norm).T
    coord_mat_norm.shape
    
    #To calculate the Covariance Matrix:
    #To get the size of the matrix:
    #m = [row, col]
    m = coord_mat_norm.shape
    sigma = (1/(m[0]))*(np.matmul(coord_mat_norm.T,coord_mat_norm))

    #To perform Single Value Decomposition (SVD) on the covariance matrix:
    U,S,V = np.linalg.svd(sigma)

    #To find the coefficients of the straight line 
    #that minimizes the projection error:
    coeff = U[:,0]
    coeff_mat = np.reshape(coeff,(1,3))

    #Validation Check:
    z_cal = np.matmul(coeff.T,coord_mat_norm.T)
    z_cal_mat = np.reshape(z_cal,(len(x),1))

    #To go back to the original:
    U_reduce_z = np.matmul(z_cal_mat, coeff_mat)
    coords_cal = []
    for k in range(len(x)):
        coords_cal.append(U_reduce_z[k,:] + mean_cols)
    coords_cal = np.array(coords_cal)

    #To plot a line through all PCA coordinates:
    #To define an initial PCA coordinate:
    p_i = coords_cal[0,:]

    #To define a scalar "t" parameter:
    #This scalar parameter can have with any integer value. 
    #The negative sign signifies the direction.
    t = -10.00*len(x)

    #To calculate the final (x_f,y_f,z_f) Coordinate 
    #such that it lies on the line:
    p_f  = []
    for i in range(len(p_i)):
        p_f.append(p_i[i] + t*coeff[i])
    p_f = np.array(p_f)
    PCA_line_coords = np.array([p_i, p_f])

    #To Calculate "t" parameters for all projected PCA coordinates:
    t = []
    for i in range(len(coords_cal[:,0])):
        t.append((coords_cal[i,0] - coords_cal[0,0])/coeff[0])

    return coords_cal, PCA_line_coords, t

def BinAndSort(x_unsorted, y_unsorted, z_unsorted, x_pca_unsorted, y_pca_unsorted, z_pca_unsorted, t, N):
    '''
    This function bins and sorts a given set of Cartesian coordinates 
    and the corresponding PCA cartesian Coorinates (including the scalar t parameter).
    
    INPUT ARUGUMENTS:
    x_unsorted = An array of orginal x Cartesian coordinates 
                 (as generated by the JetCurry Code)
    y_unsorted = An array of orginal y Cartesian coordinates 
                 (as generated by the JetCurry Code)
    z_unsorted = An array of orginal z Cartesian coordinates 
                 (as generated by the JetCurry Code)
    x_pca_unsorted = An array of x Cartesian coordinates calculated using PCA
    y_pca_unsorted = An array of y Cartesian coordinates calculated using PCA
    z_pca_unsorted = An array of z Cartesian coordinates calculated using PCA
    t = An array of scalar parameters (associated with PCA coordinates)
    N = Number of subintervals
    
    OUTPUT ARGUMENTS:
    x_binned = An array of binned x Cartesian coordinates
    y_binned = An array of binned y Cartesian coordinates
    z_binned = An array of binned z Cartesian coordinates
    x_pca_binned = An array of binned x PCA Cartesian coordinates
    y_pca_binned = An array of binned y PCA Cartesian coordinates
    z_pca_binned = An array of binned z PCA Cartesian coordinates
    t_binned = An array of binned scalar parameters (associated with binned PCA coordinates)
    
    '''
    #Arrays to store the (x,y,z) coordinates:
    #These arrays correspond to the coordinates that have been binned:
    x_binned = []
    y_binned = []
    z_binned = []
    x_pca_binned = []
    y_pca_binned = []
    z_pca_binned = []
    t_binned = []

    #To calculate the bin size (uniform):
    #To find the endpoints of "t" parameter:
    #t is the unsorted array from PCA:
    t_i = np.min(t)
    t_f = np.max(t)

    #Bin size:
    h = np.abs(t_f-t_i)/(N)
    
    #For n: Number of Datapoints
    for n in range(0,N):
        #To count the number of elements in a bin:
        index = 0

        #To initialize the temp arrays that will be used to save the coordinates:
        x_temp = []
        y_temp = []
        z_temp = []
        x_pca_temp = []
        y_pca_temp = []
        z_pca_temp = []
        t_temp = []
                
        for i in range(len(t)):
            if ((n*h) < np.abs(t[i]) < ((n+1)*h)):
                x_temp.append(x_unsorted[i])
                y_temp.append(y_unsorted[i])
                z_temp.append(z_unsorted[i])
                x_pca_temp.append(x_pca_unsorted[i])
                y_pca_temp.append(y_pca_unsorted[i])
                z_pca_temp.append(z_pca_unsorted[i])
                t_temp.append(t[i])
                index = index + 1
        if (index != 0):
            x_binned.append(np.mean(x_temp))
            y_binned.append(np.mean(y_temp))
            z_binned.append(np.mean(z_temp))
            x_pca_binned.append(np.mean(x_pca_temp))
            y_pca_binned.append(np.mean(y_pca_temp))
            z_pca_binned.append(np.mean(z_pca_temp))
            t_binned.append(np.mean(t_temp))
        else: continue
    
    #To display the following parameters:
#    print("After binning and sorting: \n")
#    print("For Cartesian Coordinates:")
#    print("Number of subintervals between binned Cartesian coordinates: %0.2f" %(len(x_binned)-1))
#    print("Number of binned Coordinates: %0.2f" %(len(x_binned)))
#    print("For PCA Cartesian Coordinates:")
#    print("Number of subintervals between binned PCA Cartesian coordinates: %0.2f" %(len(x_pca_binned)-1))
#    print("Number of binned Coordinates: %0.2f" %(len(x_pca_binned)))
    return (np.array(x_binned), np.array(y_binned), np.array(z_binned), np.array(x_pca_binned), np.array(y_pca_binned), np.array(z_pca_binned), np.array(t_binned))


def LinearSplinesPlot(x,y):
    '''
    This function calculates the coefficients of the 
    Lagrange interpolating polynomials that 
    define linear splines. 
    This function also plots linear splines.
    INPUT ARGUMENTS:
    x  = A vector of x coordinates
    y  = A vector of y coordinates
    OUTPUT ARGUMENTS:
    fig = A plot with linear splines that connect the 
          given (x,y) data points
    linear_splines_coeff = Coefficients of the linear spline functions
    '''
    #To initial an array to save the coefficients for each of the linear splines:
    linear_splines_coeff = []
    for i in range(len(y)-1):
        #Calculates the coefficients for the Lagrange Interpolating Polynomials:
        a1 = y[i]/(x[i] - x[i+1])
        a2 = y[i+1]/(x[i+1] - x[i])
        linear_splines_coeff.append([a1,a2])
        linear_splines_coeff
        f_firstorder_2D = lambda X : a1*(X-x[i+1]) + a2*(X-x[i])
        X = np.linspace(x[i],x[i+1],2)
        y_interp_interval_endpoints = [*map(f_firstorder_2D, X)]
        fig = plt.plot(X,y_interp_interval_endpoints, color = "black", linewidth= 0.8)
    return fig, linear_splines_coeff

def InterpValue1DLinearSplines(x, x_binned,coeff_sorted_array):
    '''
    This function calculates the interpolated value 
    for a given x value based on the function (spline) 
    defined in an given x interval.
    INPUT ARGUMENTS:
    x = An array of values for which the 
        interpolated value[s] are calculated  
    x_binned = An array of values that defines the start 
               and the end points of an interval. 
               These include the first point, the last point, 
               and the knots.
    FuncsArray = An array of anonymous functions where 
                 the kth function defines 
                 the kth spline polynomial between x_binned[k]
                 and x_binned[k+1] points.
    OUTPUT ARGUMENTS:
    interp_value = An array of interpolated values for 
                   the corresponding x values.
    '''
    #To initializa an array to store the interpolated values and corresponding x values:
    x_at_interp_val = []
    interp_val = []
    
    #Number of binned datapoints:
    n = len(x_binned)
    
    for i in range(len(x)):
        for k in range(len(x_binned)-1):
            if (x_binned[k+1] <= x[i] <= x_binned[k]):
                #Anonymous function to define the linear interpolating polynomial between x[i] and x[i+1]:
                f = lambda X: (coeff_sorted_array[k][0])*(X-x_binned[k+1]) + (coeff_sorted_array[k][1])*(X-x_binned[k])
                x_at_interp_val.append(x[i])
                interp_val.append(f(x[i]))
    return (np.array(x_at_interp_val), np.array(interp_val))

def InterpValue1DQuadraticSplines(x,x_binned,coeff_sorted_array):
    '''
    This function calculates the interpolated value 
    for a given x value based on the function (spline) 
    defined in an given x interval.
    INPUT ARGUMENTS:
    x = An array of values for which the 
        interpolated value[s] are calculated  
    x_binned = An array of values that defines the start 
               and the end points of an interval. 
               These include the first point, the last point, 
               and the knots.
    FuncsArray = An array of anonymous functions where 
                 the kth function defines 
                 the kth spline polynomial between x_binned[k]
                 and x_binned[k+1] points.
    OUTPUT ARGUMENTS:
    interp_value = An array of interpolated values for 
                   the corresponding x values.
    '''
    #To initialize an array to store the interpolated values:
    x_coord = []
    interp_val = []
    
    #Number of binned datapoints:
    n = len(x_binned)
    
    for i in range(len(x)):
        for k in range(len(x_binned)-1):
            if (x_binned[k+1] <= x[i] <= x_binned[k]):
                #Anonymous function to define the quadratic interpolating polynomial between x[i] and x[i+1]:
                f = lambda X: coeff_sorted_array[k]*(X**2) + coeff_sorted_array[k+n-1]*(X) + coeff_sorted_array[k+2*(n-1)]
                x_coord.append(x[i])
                interp_val.append(f(x[i]))
    return (np.array(x_coord),np.array(interp_val))

def ChiSquared(observed_values, expected_values):
    '''
    This function calculates the Chi Squared number 
    to quantify the agreement between the observed values 
    and the estimated values.
    NOTE: For this project, the observed values 
          and the estimated values 
          are the associated Cartesian coodinates.
    Input Arguments:
    observed_values = An array of the Cartesian coordinates calculated 
                      using an interpolation technique generated 
                      by the JetCurry Code.
    expected_values = An array of the Cartesian coordinates calculated 
                      using an interpolation technique.
    Output Arguments:
    ChiSquare = A number that quantifies the agreement between 
                the observed values and the estimated values. 
                This number represents the overall agreement between 
                the observed coordinates and the interpolated coordinates if 
                the observed data were to follow the interpolated model.
    '''
    #Initial value of the Chi Squared:
    ChiSquare_test=0
    
    for obs_val, exp_val in zip(observed_values, expected_values):
        ChiSquare_test+=((float(obs_val)-float(exp_val))**2)/float(exp_val)
        
    return ChiSquare_test