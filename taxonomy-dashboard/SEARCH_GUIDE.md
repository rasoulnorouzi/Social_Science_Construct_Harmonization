# Search Bar Guide

## Overview
The APA Psychological Thesaurus now includes a powerful search bar for quick term lookup alongside hierarchical browsing.

## How to Use

### Basic Search
1. Click the search input field or start typing
2. Type any part of a term name (case-insensitive)
3. Results update instantly as you type
4. Click any result to view full term details

### Search Types

#### By Term Name
- **Partial match** (substring search)
- **Case-insensitive**
- Examples:
  - `anxiety` → All terms containing "anxiety"
  - `Abandonment` → All terms containing "Abandonment"
  - `sleep` → Terms like "Insomnia", "Sleep Disorders"

#### By Subject Code
- **Exact match**
- Examples:
  - `00005` → Shows term with code 00005
  - `83640` → Shows term with that specific code

### Result Display
Each search result shows:
- **Term Name** - Main concept label
- **Subject Code** - Classification code (if available)
- **Term Type** - Badge indicating PT (Preferred Term) or ND (Non-Descriptor)

Results are automatically sorted alphabetically by term name.

### Visual Feedback
- **Blue border** - Search found results
- **Red border** - No results found
- **Result count** - Shows "X results" at top
- **Hover effect** - Results highlight on hover

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| **Typing** | Real-time search (1+ character to search) |
| **Escape** | Clear search & return to tree view |
| **Enter** | No action (prevented for UX) |

## Switching Views

### From Tree View to Search
- Click in the search box
- Start typing a term name or code
- Tree view automatically replaced with search results

### From Search Results to Tree View
- Clear the search box (delete text or press Escape)
- Tree view automatically restored

## Search Examples

### Finding Specific Terms
**Search:** `depression`
**Results:** All terms containing "depression"
- Major Depressive Disorder
- Postpartum Depression
- Treatment-Resistant Depression
- Depression (general)

**Search:** `behavior`
**Results:** All behavioral concepts
- Attachment Behavior
- Prosocial Behavior
- Problem Behavior
- Behavior Modification

### Finding by Code
**Search:** `00005`
**Result:** Single term with that exact code

### Partial Matches
**Search:** `anxiety`
**Results:** 
- Anxiety
- Anxiety Disorders
- Separation Anxiety
- Trait Anxiety
- Test Anxiety
- (+ more variations)

## Tips & Tricks

✅ **Case-insensitive** - "ANXIETY", "anxiety", "Anxiety" all work
✅ **Partial matching** - "anx" finds "anxiety"
✅ **Fast switching** - Escape clears search instantly
✅ **Sorted results** - Alphabetical order for easy scanning
✅ **Type indicators** - PT/ND badges show term status
✅ **Code lookup** - Use exact subject codes for precise searches

## Integration with Dashboard

After finding a term via search:
1. Click the search result
2. Full term details displayed in main panel
3. See:
   - Complete metadata (code, year, type)
   - Full definition/scope note
   - History note (if available)
   - All 5 relation types with weights
4. Click any relation card to navigate

## Search Performance

- Searches across **500,000+ terms** instantly
- Real-time filtering as you type
- No server calls (all client-side)
- Optimized for browser performance

## Combining Techniques

### Method 1: Search + Browse
1. Search for term by name → Find match
2. Click result to view details
3. Browse relations and navigate hierarchically

### Method 2: Search + Code
1. If you know the subject code
2. Type the code directly
3. View the term

### Method 3: Browse + Search
1. Explore tree hierarchically
2. When you find a general area
3. Use search to find specific terms within that area

## Technical Details

**Search Algorithm:**
- Case-insensitive substring matching
- Searches term names and subject codes
- O(n) complexity but very fast on 500k terms
- Results sorted alphabetically

**Search Scope:**
- Term name (primary search field)
- Subject code (secondary search field)
- Does NOT search definitions (scope notes)
- Does NOT search history notes
- Does NOT search relation data

## Troubleshooting

### Search returns no results
- Check spelling
- Try partial term name
- Use fewer characters
- Try searching by code instead

### Search is slow
- Not expected on modern browsers
- All search is client-side (very fast)
- First load takes 5-10s (XML parsing)
- Searches after that are instant

### Accidentally cleared search
- Press Escape to restore tree view
- Or just keep searching

---

**Last Updated:** February 13, 2026
