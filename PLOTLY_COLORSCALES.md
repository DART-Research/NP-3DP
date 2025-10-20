# Plotly Colorscales Reference

Copy and paste the colorscale names directly into your visualization code.

## Sequential Colorscales (Single-Hue)
Best for data that progresses from low to high values.

```
'Blues'      # Light to dark blue
'Greens'     # Light to dark green  
'Greys'      # Light to dark grey
'Oranges'    # Light to dark orange
'Purples'    # Light to dark purple
'Reds'       # Light to dark red
```

## Sequential Colorscales (Multi-Hue)
Best for data that progresses from low to high with multiple colors.

### Perceptually Uniform (Recommended)
```
'Viridis'    # Purple-blue-green-yellow, perceptually uniform
'Plasma'     # Purple-pink-yellow, perceptually uniform
'Inferno'    # Black-red-yellow, perceptually uniform
'Magma'      # Black-purple-white, perceptually uniform
'Cividis'    # Blue to yellow, colorblind-friendly
'Turbo'      # Rainbow-like but perceptually improved
```

### Temperature-Based
```
'Hot'        # Black-red-yellow-white
'Warm'       # Red to yellow
'Cool'       # Blue to cyan
'Thermal'    # Thermal/heat map colors
```

### Nature-Inspired
```
'YlGn'       # Yellow-green
'YlGnBu'     # Yellow-green-blue
'YlOrRd'     # Yellow-orange-red
'YlOrBr'     # Yellow-orange-brown
'BuGn'       # Blue-green
'BuPu'       # Blue-purple
'GnBu'       # Green-blue
'PuBu'       # Purple-blue
'PuBuGn'     # Purple-blue-green
'PuRd'       # Purple-red
'RdPu'       # Red-purple
'OrRd'       # Orange-red
```

### Other Sequential
```
'Blackbody'  # Black body radiation
'Electric'   # Electric/neon colors
'Bone'       # Black-blue-white
'Copper'     # Black to copper
'Gray'       # Light to dark gray
'Pink'       # Pink shades
'Peach'      # Light orange to red
'Mint'       # Light green to cyan
'Algae'      # Green algae-like colors
```

## Diverging Colorscales
Best for data with a meaningful midpoint (e.g., positive/negative deviations).

```
'RdBu'       # Red-white-blue (classic diverging)
'RdYlBu'     # Red-yellow-blue
'RdYlGn'     # Red-yellow-green
'Spectral'   # Red-orange-yellow-green-blue
'RdGy'       # Red-white-grey
'Coolwarm'   # Cool blue to warm red
'BWR'        # Blue-white-red
'Seismic'    # Blue-white-red (for seismic data)
'Balance'    # Brown-white-teal
'Picnic'     # Green-white-red
'Portland'   # Custom diverging scale
'Temps'      # Blue-white-red temperature scale
'Tealrose'   # Teal-white-rose
'Tropic'     # Cyan-white-magenta
'Earth'      # Brown-white-green earth tones
```

## Cyclic Colorscales
Best for periodic/circular data (e.g., angles, phases, time of day).

```
'HSV'        # Full HSV color wheel
'Phase'      # Phase angle visualization
'Twilight'   # Purple-white-purple
'IceFire'    # Blue-black-red-white-blue
'Edge'       # Edge-like cyclic scale
```

## Special Purpose
Use with caution - some may not be perceptually uniform.

```
'Rainbow'    # Traditional rainbow (avoid for scientific data)
'Jet'        # Blue-cyan-yellow-red (avoid for scientific data)
'Plotly'     # Plotly default categorical colors
'Plotly3'    # Plotly categorical colors
'Aggrnyl'    # Amber-green-nylon
'Agsunset'   # Agricultural sunset
```

---

## Usage Examples

### In Your Visualization Code

```python
# Sequential data (e.g., height, temperature)
colorscale='Viridis'

# Diverging data (e.g., profit/loss, above/below average)
colorscale='RdBu'

# Single color (all curves the same)
colorscale='#FF5733'  # Hex color

# Custom gradient
colorscale=[
    [0.0, 'rgb(0,0,255)'],    # Blue at minimum
    [0.5, 'rgb(255,255,255)'], # White at middle  
    [1.0, 'rgb(255,0,0)']      # Red at maximum
]
```

### Reversed Colorscales
Add `_r` to any colorscale name to reverse it:

```python
colorscale='Viridis_r'   # Yellow to purple (reversed)
colorscale='RdBu_r'      # Blue to red (reversed)
```

---

## Quick Recommendations

| Data Type | Recommended Colorscales | Why |
|-----------|------------------------|-----|
| **General Sequential** | `'Viridis'`, `'Plasma'`, `'Turbo'` | Perceptually uniform, works for most data |
| **Temperature/Heat** | `'Hot'`, `'Thermal'`, `'Inferno'` | Intuitive for temperature data |
| **Diverging/Difference** | `'RdBu'`, `'Coolwarm'`, `'Balance'` | Clear neutral midpoint |
| **Colorblind-Friendly** | `'Cividis'`, `'Viridis'` | Accessible to all viewers |
| **Terrain/Elevation** | `'Earth'`, `'Terrain'`, `'YlOrBr'` | Natural earth tones |
| **Single Color** | `'Blues'`, `'Greens'`, `'Reds'` | Clean monochromatic look |

---

## Tips

- **Avoid** `'Jet'` and `'Rainbow'` for scientific data - they're not perceptually uniform
- **Use diverging** colorscales only when your data has a meaningful center point
- **Use sequential** colorscales for data that goes from low to high
- **Add `_r`** to reverse any colorscale direction
- **Test** your visualization with `'Cividis'` to ensure colorblind accessibility