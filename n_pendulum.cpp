#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <format>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct PendulumParams
{
    int n;
    Vec m;
    Vec l;
    Vec suffix_mass;
    double g = 9.81;

    PendulumParams(Vec masses, Vec lengths)
        : n(masses.size()), m(std::move(masses)), l(std::move(lengths))
    {
        suffix_mass.resize(n);
        suffix_mass[n - 1] = m[n - 1];
        for (int i = n - 2; i >= 0; --i)
            suffix_mass[i] = suffix_mass[i + 1] + m[i];
    }
};

struct State
{
    Vec q;
    Vec dq;

    State(int n) : q(Vec::Zero(n)), dq(Vec::Zero(n)) {}
    State(Vec angles, Vec velocities)
        : q(std::move(angles)), dq(std::move(velocities)) {}

    State operator+(const State &o) const { return {q + o.q, dq + o.dq}; }
    State operator*(double s) const { return {q * s, dq * s}; }
    friend State operator*(double s, const State &st) { return st * s; }
};

Mat build_M(const State &s, const PendulumParams &p)
{
    Mat M(p.n, p.n);
    for (int j = 0; j < p.n; ++j)
    {
        for (int k = 0; k < p.n; ++k)
        {
            M(j, k) = std::cos(s.q[j] - s.q[k]) * p.l[j] * p.l[k] * p.suffix_mass[std::max(j, k)];
        }
    }

    return M;
}

Vec build_C(const State &s, const PendulumParams &p)
{
    Vec C = Vec::Zero(p.n);
    for (int j = 0; j < p.n; ++j)
    {
        for (int k = 0; k < p.n; ++k)
        {
            if (j == k)
                continue;

            double coeff = std::sin(s.q[j] - s.q[k]) * p.l[j] * p.l[k] * p.suffix_mass[std::max(j, k)];

            C[j] += coeff * s.dq[k] * s.dq[k];
        }
    }

    return C;
}

Vec build_G(const State &s, const PendulumParams &p)
{
    Vec G(p.n);
    for (int j = 0; j < p.n; ++j)
    {
        G[j] = p.g * p.l[j] * std::sin(s.q[j]) * p.suffix_mass[j];
    }
    return G;
}

State derivatives(const State &s, const PendulumParams &p)
{
    Mat M = build_M(s, p);
    Vec rhs = -(build_C(s, p) + build_G(s, p));

    Vec ddq = M.llt().solve(rhs);

    return State(s.dq, ddq);
}

State rk4_step(const State &s, const PendulumParams &p, double dt)
{
    State k1 = derivatives(s, p);
    State k2 = derivatives(s + 0.5 * dt * k1, p);
    State k3 = derivatives(s + 0.5 * dt * k2, p);
    State k4 = derivatives(s + dt * k3, p);

    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void simulate(const PendulumParams &p, State state,
              double dt, double T, const std::string &filename)
{
    std::ofstream out(filename);

    out << "t";
    for (int i = 0; i < p.n; i++)
        out << ",q" << i;
    for (int i = 0; i < p.n; i++)
        out << ",dq" << i;
    out << "\n";

    int steps = static_cast<int>(T / dt);
    for (int step = 0; step <= steps; ++step)
    {
        double t = step * dt;
        out << t;
        for (int i = 0; i < p.n; i++)
            out << "," << state.q[i];
        for (int i = 0; i < p.n; i++)
            out << "," << state.dq[i];
        out << "\n";
        state = rk4_step(state, p, dt);
    }
}

int main()
{
    int n = 30;
    Vec m = Vec::Ones(n);
    Vec l = Vec::Constant(n, 0.5);
    PendulumParams p(m, l);

    double totalTime = 20.0;
    double dt = 1e-3;

    int N_traj = 30;
    for (int i = 0; i < N_traj; ++i)
    {
        Vec q0 = Vec::Constant(n, 0.0);
        for (int j = 1; j < n; j++)
            q0[j] = 0.1 + q0[j-1];

        q0[0] += i * 1e-4;
        State s0(q0, Vec::Zero(n));
        auto filename = std::format("results/traj_{:03}.csv", i);
        simulate(p, s0, dt, totalTime, filename);
    }
}