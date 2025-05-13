import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey

# Create a flowchart using matplotlib with boxes and arrows

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # Define boxes with text
    boxes = {
        'start': (0.4, 0.9, 0.2, 0.05, 'Start: Input Sentence and Target Word'),
        'pos_tag': (0.4, 0.82, 0.2, 0.05, 'POS Tagging of Target Word'),
        'get_senses': (0.4, 0.74, 0.2, 0.05, 'Get WordNet Senses (Filtered by POS)'),
        'process_context': (0.4, 0.66, 0.2, 0.05, 'Process Context with Positional Weighting'),
        'check_collocations': (0.4, 0.58, 0.2, 0.05, 'Check for Collocations'),
        'apply_rules': (0.4, 0.5, 0.2, 0.05, 'Apply Rule-Based Boosting'),
        'calculate_overlap': (0.4, 0.42, 0.2, 0.05, 'Calculate Overlap Score (Lesk)'),
        'bert_similarity': (0.4, 0.34, 0.2, 0.05, 'Calculate BERT Semantic Similarity'),
        'feedback_boost': (0.4, 0.26, 0.2, 0.05, 'Apply Feedback Boost'),
        'combine_scores': (0.4, 0.18, 0.2, 0.05, 'Combine Scores with Weights'),
        'select_best': (0.4, 0.1, 0.2, 0.05, 'Select Best Sense and Alternatives'),
        'end': (0.4, 0.02, 0.2, 0.05, 'End: Return Disambiguation Result')
    }

    # Draw boxes
    for key, (x, y, w, h, text) in boxes.items():
        rect = plt.Rectangle((x, y), w, h, fill=True, edgecolor='black', facecolor='#cce5ff')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, wrap=True)

    # Draw arrows between boxes
    def draw_arrow(start_key, end_key):
        x_start, y_start, w_start, h_start, _ = boxes[start_key]
        x_end, y_end, w_end, h_end, _ = boxes[end_key]
        ax.annotate('', xy=(x_end + w_end/2, y_end + h_end), xytext=(x_start + w_start/2, y_start),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    flow_sequence = [
        'start', 'pos_tag', 'get_senses', 'process_context', 'check_collocations',
        'apply_rules', 'calculate_overlap', 'bert_similarity', 'feedback_boost',
        'combine_scores', 'select_best', 'end'
    ]

    for i in range(len(flow_sequence) - 1):
        draw_arrow(flow_sequence[i], flow_sequence[i+1])

    plt.title('Flowchart of Enhanced Lesk-based Word Sense Disambiguation Algorithm', fontsize=14)
    plt.show()

# Draw the flowchart
draw_flowchart()
