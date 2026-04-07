---
layout: default
title: Diary
permalink: /diary/
---

# Diary

Short personal entries live here rather than on the homepage.

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
