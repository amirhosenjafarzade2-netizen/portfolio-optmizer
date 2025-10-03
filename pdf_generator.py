from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

def generate_pdf_report(results, use_transaction_costs, use_taxes, use_inflation, inflation_rate):
    """Generate a PDF report of portfolio optimization results."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Portfolio Optimization Report", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    for method, result in results.items():
        elements.append(Paragraph(f"{method} Summary", styles['Heading2']))
        summary_data = [
            ["Metric", "Value"],
            ["Expected Return (Nominal)", f"{result['return'] * 100:.2f}%"],
            ["Expected Volatility", f"{result['volatility'] * 100:.2f}%"],
            ["Sharpe Ratio", f"{result['sharpe']:.2f}"],
            ["Value-at-Risk (95%)", f"{result['var'] * 100:.2f}%"],
            ["Conditional VaR (95%)", f"{result['cvar'] * 100:.2f}%"]
        ]
        if use_transaction_costs:
            summary_data.append(["Transaction Cost Impact", f"{result['cost_penalty'] * 100:.2f}%"])
        if use_taxes:
            summary_data.append(["Tax Impact", f"{result['tax_impact'] * 100:.2f}%"])
        if use_inflation:
            adjusted_return = result['return'] - result['tax_impact'] - result['cost_penalty'] - (inflation_rate if use_inflation else 0.0)
            summary_data.append(["Adjusted Return (after inflation)", f"{adjusted_return * 100:.2f}%"])
        elements.append(Table(summary_data, colWidths=[3 * inch, 2 * inch], style=[
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey)
        ]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    return pdf_buffer.getvalue()
