# mcusqueeze/analysis/compatibility.py

from typing import Dict, List
from mcusqueeze.targets import SUPPORTED_TARGETS


def check_target_compatibility(op_analysis: Dict, model_size_kb: float, memory_kb: float, target: str) -> Dict:
    """
    Check if a model is compatible with a target MCU.
    """
    
    target_info = SUPPORTED_TARGETS.get(target)
    if not target_info:
        return {
            'target': target,
            'compatible': False,
            'issues': [f"Unsupported target: '{target}'"],
            'warnings': [],
        }
    
    issues = []
    warnings = []
    
    # 1. Check operation compatibility
    supported_ops = target_info.get('supported_ops', [])
    unique_ops = op_analysis.get('unique_ops', set())
    
    unsupported_ops = []
    for op_type in unique_ops:
        if op_type not in supported_ops:
            unsupported_ops.append(op_type)
    
    if unsupported_ops:
        issues.append(f"Unsupported operations: {', '.join(unsupported_ops)}")
    
    # 2. Check flash compatibility
    flash_kb = target_info.get('flash_kb', 0)
    flash_compatible = True
    flash_usage = 0
    
    if flash_kb > 0:
        flash_usage = (model_size_kb / flash_kb) * 100
        if model_size_kb > flash_kb:
            flash_compatible = False
            issues.append(f"Model exceeds flash: {model_size_kb:.1f} KB / {flash_kb} KB")
        elif flash_usage > 80:
            warnings.append(f"Flash usage high: {flash_usage:.1f}% of {flash_kb} KB")
    
    # 3. Check RAM compatibility
    ram_kb = target_info.get('ram_kb', 0)
    ram_compatible = True
    ram_usage = 0
    
    if ram_kb > 0 and memory_kb > 0:
        ram_usage = (memory_kb / ram_kb) * 100
        if memory_kb > ram_kb:
            ram_compatible = False
            issues.append(f"Model exceeds RAM: {memory_kb:.1f} KB / {ram_kb} KB")
        elif ram_usage > 80:
            warnings.append(f"RAM usage high: {ram_usage:.1f}% of {ram_kb} KB")
    
    all_compatible = (flash_compatible and ram_compatible and len(unsupported_ops) == 0)
    
    return {
        'target': target,
        'target_info': target_info,
        'compatible': all_compatible,
        'unsupported_ops': unsupported_ops,
        'flash_kb': flash_kb,
        'ram_kb': ram_kb,
        'model_size_kb': model_size_kb,
        'memory_kb': memory_kb,
        'flash_usage': flash_usage,
        'ram_usage': ram_usage,
        'flash_compatible': flash_compatible,
        'ram_compatible': ram_compatible,
        'issues': issues,
        'warnings': warnings,
    }


def get_compatibility_summary(compat: Dict) -> str:
    """
    Get a human-readable summary of compatibility.
    """
    
    lines = []
    lines.append("🎯 Target Compatibility:")
    lines.append("=" * 40)
    
    target_info = compat.get('target_info', {})
    lines.append(f"Target: {target_info.get('name', compat['target'])}")
    lines.append(f"  Flash: {compat.get('flash_kb', 0)} KB")
    lines.append(f"  RAM:   {compat.get('ram_kb', 0)} KB")
    lines.append("")
    
    # Operation compatibility
    unsupported_ops = compat.get('unsupported_ops', [])
    if unsupported_ops:
        lines.append(f"[red]✗[/] Unsupported ops: {', '.join(unsupported_ops)}")
    else:
        lines.append("[green]✓[/] All operations supported")
    
    lines.append("")
    
    # Flash compatibility
    if compat.get('flash_compatible', False):
        lines.append(f"[green]✓[/] Flash: {compat['model_size_kb']:.1f} KB / {compat['flash_kb']} KB ({compat['flash_usage']:.1f}%)")
    else:
        lines.append(f"[red]✗[/] Flash: {compat['model_size_kb']:.1f} KB / {compat['flash_kb']} KB (EXCEEDS)")
    
    # RAM compatibility
    if compat.get('memory_kb', 0) > 0:
        if compat.get('ram_compatible', False):
            lines.append(f"[green]✓[/] RAM: {compat['memory_kb']:.1f} KB / {compat['ram_kb']} KB ({compat['ram_usage']:.1f}%)")
        else:
            lines.append(f"[red]✗[/] RAM: {compat['memory_kb']:.1f} KB / {compat['ram_kb']} KB (EXCEEDS)")
    
    lines.append("")
    
    # Overall verdict
    if compat.get('compatible', False):
        lines.append("[green]✅ Model is compatible with this target![/]")
    else:
        lines.append("[red]❌ Model is NOT compatible[/]")
        for issue in compat.get('issues', []):
            lines.append(f"    • {issue}")
    
    # Warnings
    warnings = compat.get('warnings', [])
    if warnings:
        lines.append("")
        lines.append("[yellow]⚠ Warnings:[/]")
        for warning in warnings:
            lines.append(f"    • {warning}")
    
    return "\n".join(lines)


