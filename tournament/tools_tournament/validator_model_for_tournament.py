import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import traceback
import sys
from stable_baselines3 import PPO

class ModelValidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI - Tournament Model Validator")
        self.root.geometry("700x580")
        self.root.configure(bg="#1e1e2f")
        
        # Konfigurasi Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_col = "#1e1e2f"
        fg_col = "#e0e0e0"
        btn_col = "#3a3a52"
        btn_hover = "#4a4a68"
        accent = "#00d2ff"
        
        style.configure("TFrame", background=bg_col)
        style.configure("TLabel", background=bg_col, foreground=fg_col, font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground=accent)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#8a8a8a")
        
        style.configure("TButton", 
                       font=("Segoe UI", 11, "bold"), 
                       background=btn_col, 
                       foreground=fg_col,
                       borderwidth=0,
                       focuscolor=btn_col,
                       padding=10)
        style.map("TButton", background=[("active", btn_hover)])
        
        style.configure("Val.TButton", background="#00c853", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.map("Val.TButton", background=[("active", "#00e676")])

        # Main Layout Frame
        main_frame = ttk.Frame(root, padding=25)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header Section
        ttk.Label(main_frame, text="TOURNAMENT MODEL VALIDATOR", style="Header.TLabel").pack(pady=(0, 5))
        ttk.Label(main_frame, text="Periksa kompatibilitas file model .zip dengan standar arena 1v1", style="SubHeader.TLabel").pack(pady=(0, 15))
        
        # Tombol Pilih File
        self.btn_browse = ttk.Button(main_frame, text="📂 Pilih dan Validasi Model .zip", style="Val.TButton", command=self.load_and_validate)
        self.btn_browse.pack(fill=tk.X, pady=(0, 15))
        
        # Scrolled Text untuk Log & Hasil Analisis
        self.txt_report = scrolledtext.ScrolledText(main_frame, width=80, height=20, bg="#12121a", fg="#e0e0e0", font=("Consolas", 10))
        self.txt_report.pack(fill=tk.BOTH, expand=True)
        
        # Tag Konfigurasi Warna Teks pada Report
        self.txt_report.tag_config("success", foreground="#4caf50")
        self.txt_report.tag_config("error", foreground="#ff5252")
        self.txt_report.tag_config("warning", foreground="#ffeb3b")
        self.txt_report.tag_config("info", foreground="#00bcd4")
        self.txt_report.tag_config("normal", foreground="#e0e0e0")

        # Pesan Panduan Awal
        self.log("[*] Silakan klik tombol 'Pilih dan Validasi Model .zip' di atas untuk memulai.\n", "info")

    def log(self, message, tag="normal"):
        self.txt_report.configure(state='normal')
        self.txt_report.insert(tk.END, message + "\n", tag)
        self.txt_report.configure(state='disabled')
        self.txt_report.yview(tk.END)

    def clear_log(self):
        self.txt_report.configure(state='normal')
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.configure(state='disabled')

    def load_and_validate(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Model Stable Baselines3 (.zip)",
            filetypes=[("Stable Baselines3 Model", "*.zip")]
        )
        
        if not file_path:
            return
            
        self.clear_log()
        self.log(f"[*] Mencoba memuat model dari: {file_path}", "info")
        
        try:
            # Memuat model menggunakan parameter keselamatan custom_objects[cite: 3]
            model = PPO.load(file_path, custom_objects={
                "learning_rate": 0.0, 
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            })
            
            self.log("\n[+] Model BERHASIL dimuat!", "success")
            self.log(f"    - Tipe Policy      : {type(model.policy)}")
            self.log(f"    - Observation Space: {model.observation_space}")
            self.log(f"    - Action Space     : {model.action_space}\n")
            
            # Jalankan fungsi pengecekan aturan turnamen
            self.validate_tournament_rules(model)
            
        except Exception as e:
            self.log("\n[-] GAGAL MEMUAT MODEL. Terjadi error saat proses ekstraksi file:", "error")
            self.log("-" * 65, "error")
            error_msg = traceback.format_exc()
            self.log(error_msg, "error")
            self.log("-" * 65, "error")
            self.log("[SARAN PERBAIKAN] Pastikan file berformat .zip yang valid dari Stable Baselines3 PPO dan bukan file arsip biasa.", "warning")

    def validate_tournament_rules(self, model):
        self.log("=" * 65, "info")
        self.log("🔎 ANALISIS KECOCOKAN DENGAN KETENTUAN TURNAMEN", "info")
        self.log("=" * 65 + "\n", "info")
        
        is_valid = True
        
        # 1. Validasi Observation Space (Harus (11,))
        obs_shape = model.observation_space.shape
        if obs_shape == (11,):
            self.log("✅ Observation Space : SESUAI", "success")
            self.log(f"   Dimensi terdeteksi: {obs_shape} (Standar arena 11 fitur observasi sensor & makanan).", "normal")
        else:
            self.log(f"❌ Observation Space : TIDAK SESUAI", "error")
            self.log(f"   Ditemukan {obs_shape}, namun turnamen membutuhkan format (11,).", "error")
            self.log("   💡 [SARAN PERBAIKAN]:", "warning")
            self.log("      Model Anda dilatih dengan jumlah sensor/fitur lingkungan yang berbeda.", "warning")
            self.log("      Sesuaikan fungsi ekstraksi observasi di lingkungan training Anda agar menghasilkan vektor berukuran 11 elemen, atau sesuaikan skrip arena utama.", "warning")
            is_valid = False
            
        self.log("")

        # 2. Validasi Action Space (Harus Discrete(4))
        try:
            act_type = model.action_space.__class__.__name__
            act_n = getattr(model.action_space, 'n', None)
            
            if act_type == "Discrete" and act_n == 4:
                self.log("✅ Action Space      : SESUAI", "success")
                self.log(f"   Tipe terdeteksi   : Discrete({act_n}) (Mendukung 4 arah navigasi mutlak).", "normal")
            else:
                self.log(f"❌ Action Space      : TIDAK SESUAI", "error")
                self.log(f"   Ditemukan {model.action_space}, namun turnamen membutuhkan ruang aksi Discrete(4).", "error")
                self.log("   💡 [SARAN PERBAIKAN]:", "warning")
                self.log("      Pastikan saat melatih model, `action_space` di-set menggunakan `spaces.Discrete(4)` untuk merepresentasikan arah (Atas, Kanan, Bawah, Kiri).", "warning")
                is_valid = False
        except Exception as e:
            self.log(f"❌ Action Space      : ERROR saat membaca tipe ruang aksi ({str(e)})", "error")
            is_valid = False

        # 3. Kesimpulan Akhir
        self.log("\n" + "=" * 65, "info")
        if is_valid:
            self.log("🎉 KESIMPULAN: MODEL SIAP DIGUNAKAN DALAM TURNAMEN!", "success")
            self.log("Model ini kompatibel sepenuhnya dengan spesifikasi sistem `tournament_app.py`.", "success")
        else:
            self.log("🛑 KESIMPULAN: MODEL BELUM SESUAI STANDAR TURNAMEN!", "error")
            self.log("Model berisiko mengalami mismatch dimensi tensor atau gagal memprediksi langkah saat di-load ke dalam arena.", "warning")
        self.log("=" * 65, "info")

if __name__ == '__main__':
    root = tk.Tk()
    app = ModelValidatorApp(root)
    root.mainloop()