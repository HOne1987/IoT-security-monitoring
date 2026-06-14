import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stats = pd.read_csv('lepotato_stats.txt', sep='\t')

cpu   = stats['cpu_pct'].values
ram   = stats['mem_mb'].values
xs    = stats['sample'].values
n     = len(stats)
cpu_mean = cpu.mean()
cpu_max  = cpu.max()
ram_mean = ram.mean()
RAM_LIMIT = 2048

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Edge Agent Operational Stability — ARM Cortex-A53 Le Potato\n'
    f'(n={n} samples, 5-minute continuous profile)',
    fontsize=13, fontweight='bold',
)

# ── Left: CPU time series ─────────────────────────────────────────────────────
ax = axes[0]
ax.plot(xs, cpu, color='#2171b5', linewidth=1.5)
ax.fill_between(xs, cpu, alpha=0.15, color='#2171b5')
ax.axhline(cpu_mean, color='#2171b5', linestyle='--', linewidth=1.5,
           label=f'Mean {cpu_mean:.2f}%')
ax.axhline(cpu_max,  color='#2171b5', linestyle=':',  linewidth=1.2,
           label=f'Max {cpu_max:.2f}%')
ax.set_title('CPU Usage Over Time', fontsize=11, fontweight='bold')
ax.set_xlabel('Sample (× 5 s → 5-minute window)', fontsize=10)
ax.set_ylabel('CPU (% of one core)', fontsize=10)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── Right: RAM time series ────────────────────────────────────────────────────
ax = axes[1]
ax.plot(xs, ram, color='#2171b5', linewidth=1.5)
ax.fill_between(xs, ram, alpha=0.15, color='#2171b5')
ax.axhline(ram_mean,  color='#2171b5', linestyle='--', linewidth=1.5,
           label=f'Mean {ram_mean:.0f} MiB')
ax.axhline(RAM_LIMIT, color='gray',    linestyle=':',  linewidth=1.2,
           label='2 GB device limit')
ax.set_title('RAM Usage Over Time', fontsize=11, fontweight='bold')
ax.set_xlabel('Sample (× 5 s)', fontsize=10)
ax.set_ylabel('RAM — RSS (MiB)', fontsize=10)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(0, RAM_LIMIT + 200)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('resource_profile_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: resource_profile_comparison.png")
