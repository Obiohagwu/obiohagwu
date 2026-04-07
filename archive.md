---
layout: default
title: Archive
permalink: /archive/
---

# Archive

## Research

{% assign research_posts = site.posts | where: "kind", "research" %}
<ul class="post-list">
  {% for post in research_posts %}
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
  {% for post in essay_posts %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h3>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
    </li>
  {% endfor %}
</ul>

## Diary

{% assign diary_posts = site.posts | where: "kind", "diary" %}
<ul class="post-list">
  {% for post in diary_posts %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <h3>
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
      </h3>
    </li>
  {% endfor %}
</ul>
