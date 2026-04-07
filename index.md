---
layout: default
title: obi ohagwu
permalink: /
---

# Obi Ohagwu

I work on mechanistic interpretability, representation geometry, and sequence and speech modeling. This site is a research log for experiments, reports, technical notes, and a smaller amount of personal writing.

The homepage foregrounds research work. Diary entries live on a separate page. If you want the shortest path into the site, start with [Best Work]({{ '/best-work/' | relative_url }}). The [Archive]({{ '/archive/' | relative_url }}) includes everything.

## Best Work

- [Your SAE looks solved. Your model disagrees]({{ '/2026/02/09/Your-SAE-looks-solved.-Your-model-disagrees.html' | relative_url }})  
  A concrete interpretability result about why reconstruction metrics can overstate behavioral faithfulness.
- [Transport Geometry as a Testable Hypothesis?]({{ '/2026/02/22/Transport-Geometry-as-a-Testable-Hypothesis.html' | relative_url }})  
  A follow-up that tightens the claim, reports what disappeared with more training, and lays out the next diagnostic step.
- [Completed 12k Run: Early Speech Emergence in a 21.9M OLMo-Hybrid Speech LM]({{ '/2026/03/14/Completed-12k-Run-Early-Speech-Emergence-in-a-21.9M-OLMo-Hybrid-Speech-LM.html' | relative_url }})  
  A speech-modeling report with metrics, samples, and a narrow viability claim.
- [OLMo-hybrid speech technical report (PDF)]({{ '/images/olmo-hybrid-speech/final_report.pdf' | relative_url }})  
  The compact artifact version of the speech result.
- [olmo-hybrid-speech code](https://github.com/Obiohagwu/olmo-hybrid-speech)  
  Code for the current speech modeling line.

## Recent Research

{% assign research_posts = site.posts | where: "kind", "research" %}
<ul class="post-list">
  {% for post in research_posts limit: 8 %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h3>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
    </li>
  {% endfor %}
</ul>

## Essays And Notes

{% assign essay_posts = site.posts | where: "kind", "essay" %}
<ul class="post-list">
  {% for post in essay_posts limit: 6 %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h3>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
    </li>
  {% endfor %}
</ul>

Diary entries are collected on the [Diary]({{ '/diary/' | relative_url }}) page.
