# APA Psychological Thesaurus Browser

## Overview

An interactive, browser-based taxonomy explorer for the **APA Thesaurus of Psychological Index Terms** (zthes10). This dashboard displays ~500,000+ psychological concepts with complete metadata including definitions, codes, introduction years, and comprehensive term relationships.

### Complete Data Coverage
✅ **Definitions** - Scope notes with full context  
✅ **Subject Codes** - Standardized codes (e.g., "00005")  
✅ **Introduction Years** - Term addition dates  
✅ **Update Status** - Track modifications  
✅ **5 Relation Types** - All hierarchical and semantic links  
✅ **Relation Weights** - Strength indicators (100=full strength)

---

## Features

### 🌳 Hierarchical Tree Navigation
- Browse root concepts down through narrower terms
- Expand/collapse with smooth animations
- Lazy-loaded children for performance
- Shows only "Preferred Terms" (PT) as root nodes

### 📊 Complete Metadata Display
- **Term Name & Type** - Preferred Term (PT) or Non-Descriptor (ND)
- **Subject Code** - Classification code
- **Introduction Year** - When added to thesaurus
- **Update Status** - "add" or "modify"
- **Vocabulary** - Assignment (APA Main Thesaurus)

### 📖 Full Definition Section
- **Scope Notes** - Complete definitions and context
- Example: "Loneliness, anxiety, and emotional...loss of support resulting from desertion or neglect. Limited to human populations."

### 🔗 All 5 Relation Types
1. **🔤 Used For** - Synonyms & alternate names (UF)
2. **🔼 Broader Terms** - Parent/general concepts (BT)
3. **🔽 Narrower Terms** - Child/specific concepts (NT)
4. **🔗 Related Terms** - Semantic connections (RT)
5. **🔄 Use Instead Of** - Preferred term mappings (USE)

Each relation shows its weight (100 = full strength).

### 🎯 Interactive Navigation
- Click related term cards to jump to that concept
- Active term highlighted in sidebar
- Instant navigation through relationship network

### 📱 Responsive Design
- **Desktop**: 320px sidebar + full content panel
- **Tablet**: Stacked layout with scrolling
- **Mobile**: Full-width optimized interface

---

## Quick Start

### Python (Recommended)
```bash
cd /home/rass/Desktop/SocialScience-ConceptIntegration
python3 -m http.server 8000
```
Open: **http://localhost:8000/taxonomy-dashboard/**

### Node.js
```bash
cd /home/rass/Desktop/SocialScience-ConceptIntegration
npx http-server -p 8000 -c-1
```

### PHP
```bash
cd /home/rass/Desktop/SocialScience-ConceptIntegration
php -S localhost:8000
```

---

## How to Use

1. **Start Server** → Run command above from workspace root
2. **Open Dashboard** → http://localhost:8000/taxonomy-dashboard/
3. **Wait for Load** → XML parsing (5-10s first time, instant after)
4. **Browse Terms** → Left sidebar shows root concepts
5. **Expand Branches** → Click ▶ to see narrower/child terms
6. **View Details** → Click term name to see metadata
7. **Navigate** → Click related term cards to explore

### Term Display Example
```
📚 Abandonment
├─ [Preferred Term] Code: 00005 | Introduced: 1997

📖 Definition:
Loneliness, anxiety, and emotional and psychological 
loss of support resulting from desertion or neglect. 
Limited to human populations.

🔤 Used For (Synonyms):
   • Desertion                          [weight: 100]
   • Abandonment Issues                 [weight: 100]

🔼 Broader Terms:
   • Attachment Behavior                [weight: 100]

🔽 Narrower Terms:
   • Parental Abandonment               [weight: 100]
   • Maternal Rejection                 [weight: 100]

🔗 Related Terms:
   • Child Abuse                        [weight: 100]
   • Separation Anxiety                 [weight: 100]
   • Loneliness                         [weight: 100]
```

---

## Technical Stack

| Component | Details |
|-----------|---------|
| **Frontend** | HTML5 + CSS3 + JavaScript (ES6+) |
| **Data Parsing** | Client-side DOMParser API |
| **Dependencies** | None (pure vanilla) |
| **Server** | Python/Node/PHP SimpleHTTPServer |
| **Performance** | In-memory cache + lazy loading |

---

## Architecture

### Data Processing
- **Source XML**: ~17MB, 494k lines, 500k+ terms
- **Parsing**: In-browser on first load (5-10 seconds)
- **Storage**: In-memory JavaScript dictionary (~50-100MB)
- **Lookups**: O(1) by term ID, instant navigation

### Fallback Path Resolution
Automatically tries:
1. `../datasets/raw_datasets/APA Thesaurus...xml`
2. `../../datasets/raw_datasets/APA Thesaurus...xml`
3. `/datasets/raw_datasets/APA Thesaurus...xml`
4. `/<filename>.xml`

Check browser console (F12) for path resolution logs.

---

## File Structure

```
taxonomy-dashboard/
├── index.html          # Complete application (single file)
└── README.md          # This documentation
```

Data loaded from:
`../datasets/raw_datasets/APA Thesaurus of Psychological Index Terms_zthes10_February 4th 2026.xml`

---

## Browser Support

✅ **Works with:**
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Requires:** ES6+, DOMParser, Fetch API, CSS Grid/Flexbox

---

## Troubleshooting

### "Error: XML file not found"
→ Run server from workspace root:
```bash
cd /home/rass/Desktop/SocialScience-ConceptIntegration
python3 -m http.server 8000
```

### Slow First Load
→ Normal (parsing 500k terms takes 5-10s)
→ Subsequent loads instant (cached)
→ Check console (F12) for progress

### Blank Page
→ Open F12 Developer Tools
→ Check Console for error messages
→ Check Network tab for XML download
→ Verify file: datasets/raw_datasets/APA*.xml

### Cache Issues
→ Clear: **Shift+Ctrl+Delete** (Windows) or **Shift+Cmd+Delete** (Mac)

---

## Data Schema

### XML Structure
```xml
<term>
  <termId>7692726</termId>
  <termUpdate>add</termUpdate>
  <termName>Abandonment</termName>
  <termType>PT|ND</termType>
  <termVocabulary>APA Main Thesaurus</termVocabulary>
  
  <termNote label="Subject Code">00005</termNote>
  <termNote label="Scope Note">Definition text...</termNote>
  <termNote label="Introduced">1997</termNote>
  
  <relation weight="100">
    <relationType>UF|BT|NT|RT|USE</relationType>
    <termId>7695109</termId>
    <termName>Desertion</termName>
  </relation>
</term>
```

### Codes
- **PT** = Preferred Term
- **ND** = Non-Descriptor (deprecated)
- **UF** = Used For
- **BT** = Broader Term
- **NT** = Narrower Term
- **RT** = Related Term
- **USE** = Use This Term Instead

---

## Attribution

**Data Source:**  
American Psychological Association  
APA Thesaurus of Psychological Index Terms (zthes10, Feb 2026)

**Project:**  
SocialScience-ConceptIntegration

---

**Last Updated:** February 2026
