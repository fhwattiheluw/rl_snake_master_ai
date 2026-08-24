# 🐍 Snake AI Tournament - Evaluation Tools

Repositori ini berisi *tools* pendukung untuk menguji dan memvalidasi model Reinforcement Learning (Stable Baselines3 PPO) sebelum model tersebut ditandingkan dalam **Snake AI Tournament 1v1**. 

Terdapat dua aplikasi utama yang disediakan untuk membantu peserta (mahasiswa) melakukan *debugging* dan evaluasi performa agen AI mereka secara mandiri:

---

## 1. validator_model_for_tournament.py

Aplikasi ini berfungsi sebagai gerbang pengecekan (prafillter) untuk memastikan arsitektur jaringan dari model `.zip` yang telah Anda latih (*train*) memenuhi standar spesifikasi arena turnamen.

## 2. solo_tester_model_tournament_app.py
Jika model Anda sudah lulus validasi dimensi, aplikasi ini digunakan untuk melakukan simulasi visual Single-Player.