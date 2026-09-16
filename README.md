# Workplace Intelligence & Personalization Platform

An AI-powered workplace room intelligence system that combines deterministic resource allocation with machine learning-based personalization to recommend the most suitable meeting room for each employee.

## Overview

Modern workplaces have limited meeting-room resources, while employees have different preferences regarding room type, floor, timing, capacity, and flexibility.

This project addresses the problem using a two-stage decision system:

1. A deterministic allocation engine handles hard constraints such as room capacity, availability, room type, floor preference, and time flexibility.
2. A machine learning ranking model evaluates multiple feasible candidates and selects the candidate that best matches the employee's preferences while considering workplace resource intelligence.

The system is designed around the principle:

> First determine what is feasible. Then use ML to determine what is preferable.

---

## Problem Statement

In a workplace with many employees and limited meeting rooms, simply finding an available room does not guarantee that the allocation is suitable for the employee.

The system needs to consider:

- Employee room-type preferences
- Preferred floors
- Preferred meeting times
- Typical meeting capacity
- Meeting duration
- Time flexibility
- Historical employee behavior
- Room demand and utilization
- Availability of alternative rooms

The goal is to personalize room allocation without violating operational constraints.

---

## Project Architecture

```text
Employee Request
       ↓
Deterministic Allocation Engine
       ↓
Hard Constraints & Feasibility
       ↓
Feasible Room Candidates
       ↓
 ┌───────────────────────┐
 │                       │
 │  1 Candidate           │ → Direct Allocation
 │                       │
 │  2+ Candidates         │ → ML Ranking
 │                       │
 └───────────────────────┘
              ↓
      CatBoost Ranking Model
              ↓
       Best Candidate
              ↓
       Room Recommendation
