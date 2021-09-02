---
layout: splash
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/header_unsplash_low-res.jpg
excerpt: "Finance - Sports Analytics - Japanese - Travel"
---

The personal webpage & blog of Ben Pollock - Toronto

<h3 class="archive__subtitle">{{ site.data.ui-text[site.locale].recent_posts | default: "Recent Posts" }}</h3>

{% if paginator %}
  {% assign posts = paginator.posts %}
{% else %}
  {% assign posts = site.posts %}
{% endif %}

{% assign entries_layout = page.entries_layout | default: 'list' %}
<div class="entries-{{ entries_layout }}">
  {% for post in posts %}
    {% include archive-single.html type=entries_layout %}
  {% endfor %}
</div>

{% include paginator.html %}