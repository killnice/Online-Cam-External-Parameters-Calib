"""np_lie.py: Implements basic SO3 and SE3 group operations.

These implementations follow A Micro Lie Theory for State Estimation in
Robotics by Joan Sola (https://arxiv.org/pdf/1812.01537.pdf) [1]

Find references to the relevant equations in function comments.

Izzy Brand, 2022
"""

import numpy as np

###############################################################################
# SO3
###############################################################################

def SO3_hat(w):
    # [1] Example 3
    wx, wy, wz = w
    S = np.array([[0, -wz, wy],
                  [wz, 0, -wx],
                  [-wy, wx, 0]])
    return S

def SO3_vee(S):
    # [1] Example 3
    w = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return w

def SO3_Exp(S):
    # [1] Example 4
    w = SO3_vee(S)
    R = SO3_exp(w)
    return R

def SO3_exp(w):
    # [1] Example 4
    θ = np.linalg.norm(w)
    S = SO3_hat(w)
    R = np.eye(3) \
      + S * np.sin(θ) / θ\
      + S @ S * (1.0 - np.cos(θ)) / θ**2
    return R

def SO3_Log(R):
    # [1] Example 4
    θ = np.arccos(0.5 * (np.trace(R) - 1.0))
    S = θ * (R - R.T) / (2.0 * np.sin(θ))
    return S

def SO3_log(R):
    # [1] Example 4
    S = SO3_Log(R)
    w = SO3_vee(S)
    return w

def SO3_left_jacobian(w):
    # [1] Equation (145), (174)
    θ = np.linalg.norm(w)
    S = SO3_hat(w)
    J_l = np.eye(3) \
        + S * (1.0 - np.cos(θ)) / θ**2 \
        + S @ S * (θ - np.sin(θ)) / θ**3
    return J_l

def SO3_left_jacobian_inverse(w):
    # [1] Equation (146)
    θ = np.linalg.norm(w)
    S = SO3_hat(w)
    J_l_inv = np.eye(3) \
            - S / 2.0 \
            + S @ S * (1.0 / θ**2 - (1.0 + np.cos(θ))/(2.0 * θ * np.sin(θ)))
    return J_l_inv

###############################################################################
# SE3
###############################################################################

def SE3_hat(τ):
    # [1] Equation (169)
    w, v = np.split(τ, 2)
    S = SO3_hat(w)
    τˆ = np.zeros([4, 4])
    τˆ[:3, :3] = S
    τˆ[:3, 3] = v
    return τˆ

def SE3_vee(τˆ):
    # [1] Equation (169)
    S = τˆ[:3, :3]
    v = τˆ[:3, 3]
    w = SO3_vee(S)
    τ = np.concatenate([w, v])
    return τ

def SE3_Exp(τˆ):
    # [1] Equation (172), (174)
    τ = SE3_vee(τˆ)
    T = SE3_exp(τ)
    return T

def SE3_exp(τ):
    # [1] Equation (172), (174)
    w, v = np.split(τ, 2)
    R = SO3_exp(w)
    J_l = SO3_left_jacobian(w)
    t = J_l @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def SE3_Log(T):
    # [1] Equation (173), (146)
    τ = SE3_log(T)
    τˆ = SE3_hat(τ)
    return τˆ

def SE3_log(T):
    # [1] Equation (173), (146)
    R = T[:3, :3]
    t = T[:3, 3]
    w = SO3_log(R)
    J_l_inv = SO3_left_jacobian_inverse(w)
    v = J_l_inv @ t
    τ = np.concatenate([w, v])
    return τ

def SE3_hat_vee():
        τ = np.random.randn(6)
        return np.allclose(τ, SE3_vee(SE3_hat(τ)))

if __name__ == '__main__':
    out = SE3_hat_vee()
    print(out)