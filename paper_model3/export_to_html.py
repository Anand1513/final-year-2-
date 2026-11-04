import html
import re
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(base_dir, 'AgriSense_Model3_Research_Paper.md')
html_path = os.path.join(base_dir, 'AgriSense_Model3_Research_Paper.html')

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

html_out = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AgriSense 3.0: Research Paper</title>
<style>
body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    line-height: 1.6; 
    max-width: 960px; 
    margin: 40px auto; 
    padding: 0 24px; 
    color: #24292e; 
    background-color: #ffffff;
}
table { 
    border-collapse: collapse; 
    width: 100%; 
    margin: 24px 0; 
    font-size: 0.88em; 
}
th, td { 
    border: 1px solid #dfe2e5; 
    padding: 10px 14px; 
    text-align: left; 
}
th { 
    background-color: #f6f8fa; 
    font-weight: 600; 
    color: #24292e;
}
tr:nth-child(even) { 
    background-color: #fcfcfc; 
}
h1, h2, h3, h4 { 
    color: #1a1f2c; 
    margin-top: 28px; 
    margin-bottom: 12px;
}
h1 { 
    font-size: 2.1em; 
    border-bottom: 2px solid #eaecef; 
    padding-bottom: 0.3em; 
}
h2 { 
    font-size: 1.55em; 
    border-bottom: 1px solid #eaecef; 
    padding-bottom: 0.3em; 
}
h3 { 
    font-size: 1.25em; 
}
h4 { 
    font-size: 1.05em; 
}
blockquote { 
    border-left: 4px solid #0969da; 
    padding: 8px 16px; 
    color: #57606a; 
    background: #f6f8fa;
    border-radius: 0 6px 6px 0;
    margin: 16px 0; 
}
code { 
    background: #f6f8fa; 
    padding: 2px 6px; 
    border-radius: 4px; 
    font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace; 
    font-size: 88%; 
}
pre { 
    background: #f6f8fa; 
    padding: 16px; 
    border-radius: 6px; 
    overflow-x: auto; 
    font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace; 
    font-size: 85%; 
    line-height: 1.45; 
    border: 1px solid #e1e4e8;
}
img { 
    max-width: 100%; 
    height: auto; 
    display: block; 
    margin: 20px auto; 
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
hr { 
    border: none; 
    border-top: 1px solid #eaecef; 
    margin: 32px 0; 
}
</style>
</head>
<body>
"""

in_pre = False
in_table = False

for line in text.split('\n'):
    if line.startswith('```'):
        if in_pre:
            html_out += '</pre>\n'
            in_pre = False
        else:
            html_out += '<pre>'
            in_pre = True
        continue
    if in_pre:
        html_out += html.escape(line) + '\n'
        continue
    
    line_s = line.strip()
    if not line_s:
        if in_table:
            html_out += '</tbody></table>\n'
            in_table = False
        continue

    if line.startswith('# '):
        html_out += f'<h1>{html.escape(line[2:])}</h1>\n'
    elif line.startswith('## '):
        html_out += f'<h2>{html.escape(line[3:])}</h2>\n'
    elif line.startswith('### '):
        html_out += f'<h3>{html.escape(line[4:])}</h3>\n'
    elif line.startswith('#### '):
        html_out += f'<h4>{html.escape(line[5:])}</h4>\n'
    elif line.startswith('> '):
        html_out += f'<blockquote>{html.escape(line[2:])}</blockquote>\n'
    elif line.startswith('---'):
        html_out += '<hr>\n'
    elif line.startswith('|') and line.endswith('|'):
        if '---' in line:
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if not in_table:
            html_out += '<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cells) + '</tr></thead><tbody>\n'
            in_table = True
        else:
            html_out += '<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in cells) + '</tr>\n'
    else:
        if in_table:
            html_out += '</tbody></table>\n'
            in_table = False
        if line.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                html_out += f'<img alt="{html.escape(m.group(1))}" src="{html.escape(m.group(2))}"><br>\n'
            else:
                html_out += f'<p>{html.escape(line)}</p>\n'
        else:
            html_out += f'<p>{html.escape(line)}</p>\n'

if in_table:
    html_out += '</tbody></table>\n'
if in_pre:
    html_out += '</pre>\n'

html_out += '</body></html>'

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_out)

print("Export completed:", html_path)

