import numpy as np

# 常量定义
GAMMA = 1.4  # 比热比

def compute_flux(rho, u, p):
    """
    计算欧拉方程的通量
    :param rho: 密度
    :param u: 速度
    :param p: 压力
    :return: 对应的质量、动量和能量通量
    """
    # 计算总能量
    E = p / (GAMMA - 1) + 0.5 * rho * u**2
    # 计算通量
    mass_flux = rho * u
    momentum_flux = rho * u**2 + p
    energy_flux = u * (E + p)
    return mass_flux, momentum_flux, energy_flux

def ausm_flux(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R):
    """
    使用 AUSM 方法计算数值通量
    :param rho_L: 左侧密度
    :param u_L: 左侧速度
    :param p_L: 左侧压力
    :param a_L: 左侧声速
    :param rho_R: 右侧密度
    :param u_R: 右侧速度
    :param p_R: 右侧压力
    :param a_R: 右侧声速
    :return: AUSM 通量
    """
    # 计算马赫数
    M_L = u_L / a_L
    M_R = u_R / a_R

    # 马赫数分裂
    M_plus = 0.5 * (M_L + abs(M_L))
    M_minus = 0.5 * (M_R - abs(M_R))

    # 计算对流通量系数
    mass_flux = 0.5 * (rho_L * a_L * M_plus + rho_R * a_R * M_minus)

    # 压力通量分裂
    P_plus = 0.5 * p_L * (1 + np.sign(M_L))
    P_minus = 0.5 * p_R * (1 - np.sign(M_R))

    # AUSM 总通量
    F_mass = mass_flux
    F_momentum = F_mass * np.where(M_plus > 0, u_L, u_R) + (P_plus + P_minus)
    F_energy = F_mass * np.where(M_plus > 0, 0.5 * u_L**2, 0.5 * u_R**2)
    return F_mass, F_momentum, F_energy

def solve_1d_euler(nx, nt, dx, dt, rho, u, p):
    """
    使用 AUSM 方法求解一维欧拉方程
    :param nx: 网格数量
    :param nt: 时间步数
    :param dx: 空间步长
    :param dt: 时间步长
    :param rho: 初始密度分布
    :param u: 初始速度分布
    :param p: 初始压力分布
    :return: 密度、速度和压力随时间演化结果
    """
    for n in range(nt):
        # 声速
        a = np.sqrt(GAMMA * p / rho)

        # 左、右状态
        rho_L, rho_R = rho[:-1], rho[1:]
        u_L, u_R = u[:-1], u[1:]
        p_L, p_R = p[:-1], p[1:]
        a_L, a_R = a[:-1], a[1:]

        # 计算数值通量
        F_mass, F_momentum, F_energy = ausm_flux(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R)

        # 更新守恒变量
        mass = rho * u
        momentum = rho * u**2 + p
        energy = p / (GAMMA - 1) + 0.5 * rho * u**2

        mass[1:-1] -= dt / dx * (F_mass[1:] - F_mass[:-1])
        momentum[1:-1] -= dt / dx * (F_momentum[1:] - F_momentum[:-1])
        energy[1:-1] -= dt / dx * (F_energy[1:] - F_energy[:-1])

        # 更新物理量
        rho[1:-1] = mass[1:-1] / u[1:-1]
        u[1:-1] = momentum[1:-1] / mass[1:-1]
        p[1:-1] = (GAMMA - 1) * (energy[1:-1] - 0.5 * rho[1:-1] * u[1:-1]**2)

    return rho, u, p

# 初始条件
nx = 100  # 网格数量
nt = 200  # 时间步数
dx = 1.0 / nx
dt = 0.001

# 初始密度、速度和压力分布
rho = np.ones(nx)
rho[int(nx/2):] = 0.125
u = np.zeros(nx)
p = np.ones(nx)
p[int(nx/2):] = 0.1

# 求解
rho, u, p = solve_1d_euler(nx, nt, dx, dt, rho, u, p)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(rho, label='Density')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(u, label='Velocity')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(p, label='Pressure')
plt.legend()
plt.tight_layout()
plt.savefig('testausm+++.jpg',dpi=600)
