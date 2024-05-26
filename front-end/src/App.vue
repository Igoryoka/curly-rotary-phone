<template>
  <v-app>
    <v-container>
      <v-card class="mx-auto my-12" max-width="600">
        <v-card-title>
          <span class="headline">Upload Audio or Video File</span>
        </v-card-title>
        <v-card-text>
          <v-file-input
            v-model="file"
            label="Select file"
            outlined
            dense
            @change="handleFileChange"
          ></v-file-input>
          <audio v-if="audioSrc" :src="audioSrc" controls class="mt-4 styled-audio"></audio>
          <v-btn
            color="primary"
            @click="uploadFile"
            :disabled="!file || loading"
            class="mt-4"
          >
            <v-progress-circular
              v-if="loading"
              indeterminate
              size="20"
              color="white"
            ></v-progress-circular>
            <span v-else>Upload</span>
          </v-btn>
        </v-card-text>
        <v-card-actions>
          <canvas ref="canvas" class="w-100"></canvas>
        </v-card-actions>
      </v-card>
      <v-snackbar
        v-model="showError"
        :timeout="3000"
        color="red"
        top
      >
        {{ errorMessage }}
      </v-snackbar>
      <v-snackbar
        v-model="showSuccess"
        :timeout="3000"
        color="green"
        top
      >
        {{ successMessage }}
      </v-snackbar>
    </v-container>
  </v-app>
</template>

<script>
import axios from 'axios';
import Chart from 'chart.js/auto';

export default {
  data() {
    return {
      file: null,
      emotions: [],
      errorMessage: '',
      successMessage: '',
      loading: false,
      audioSrc: null,
      showError: false,
      showSuccess: false,
      chart: null
    };
  },
  mounted() {
    this.plotDefaultEmotions();
  },
  watch: {
    file(newFile) {
      if (!newFile) {
        this.audioSrc = null;
        this.clearChart();
        this.plotDefaultEmotions();
      }
    }
  },
  methods: {
    handleFileChange(event) {
      this.clearError();
      this.clearSuccess();

      const file = event.target.files[0];

      if (!file) {
        this.file = null;
        this.errorMessage = 'Please select a file.';
        this.showError = true;
        return;
      }

      const allowedExtensions = ['wav', 'mp3', 'mp4', 'avi', 'mov'];
      const extension = file.name.split('.').pop().toLowerCase();
      if (!allowedExtensions.includes(extension)) {
        this.file = null;
        this.errorMessage = 'Unsupported file type. Please select an audio or video file.';
        this.showError = true;
        return;
      }

      this.file = file;
      if (['wav', 'mp3'].includes(extension)) {
        this.audioSrc = URL.createObjectURL(file);
      } else {
        this.audioSrc = null;
      }
    },
    clearError() {
      this.errorMessage = '';
      this.showError = false;
    },
    clearSuccess() {
      this.successMessage = '';
      this.showSuccess = false;
    },
    clearChart() {
      if (this.chart) {
        this.chart.destroy();
        this.chart = null;
      }
    },
    async uploadFile() {
      this.clearError();
      this.clearSuccess();

      if (!this.file) {
        this.errorMessage = 'Please select a file first';
        this.showError = true;
        return;
      }

      const formData = new FormData();
      formData.append('file', this.file);
      this.loading = true;

      try {
        const response = await axios.post('http://localhost:5000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        this.emotions = response.data.emotions;
        this.plotEmotions();
        this.successMessage = 'File uploaded successfully!';
        this.showSuccess = true;
      } catch (error) {
        if (error.response) {
          this.errorMessage = `Error: ${error.response.data.error}`;
        } else {
          this.errorMessage = 'Error: Unable to process request';
        }
        this.showError = true;
        console.error(error);
      } finally {
        this.loading = false;
      }
    },
    plotDefaultEmotions() {
      const defaultEmotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful'];
      const defaultTimePoints = defaultEmotions.map((_, index) => index * 5);

      const emotionsList = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'];
      const emotionToNumber = emotionsList.reduce((acc, emotion, index) => {
        acc[emotion] = index;
        return acc;
      }, {});
      const numericSequence = defaultEmotions.map(emotion => emotionToNumber[emotion]);

      const ctx = this.$refs.canvas.getContext('2d');
      this.chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: defaultTimePoints,
          datasets: [{
            label: 'Emotions over Time',
            data: numericSequence,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
            tension: 0.1
          }]
        },
        options: {
          scales: {
            x: {
              title: {
                display: true,
                text: 'Time (s)'
              }
            },
            y: {
              ticks: {
                callback: function(value) {
                  return emotionsList[value];
                }
              },
              title: {
                display: true,
                text: 'Emotion'
              },
              min: 0,
              max: emotionsList.length - 1
            }
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: function(context) {
                  return emotionsList[context.parsed.y];
                }
              }
            }
          }
        }
      });
    },
    plotEmotions() {
      this.clearChart();

      const emotionsList = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'];
      const emotionToNumber = emotionsList.reduce((acc, emotion, index) => {
        acc[emotion] = index;
        return acc;
      }, {});
      const numericSequence = this.emotions.map(emotion => emotionToNumber[emotion]);
      const timePoints = numericSequence.map((_, index) => index * 5);

      const ctx = this.$refs.canvas.getContext('2d');
      this.chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: timePoints,
          datasets: [{
            label: 'Emotions over Time',
            data: numericSequence,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
            tension: 0.1
          }]
        },
        options: {
          scales: {
            x: {
              title: {
                display: true,
                text: 'Time (s)'
              }
            },
            y: {
              ticks: {
                callback: function(value) {
                  return emotionsList[value];
                }
              },
              title: {
                display: true,
                text: 'Emotion'
              },
              min: 0,
              max: emotionsList.length - 1
            }
          },
          plugins: {
            tooltip: {
              callbacks: {
                label: function(context) {
                  return emotionsList[context.parsed.y];
                }
              }
            }
          }
        }
      });
    }
  }
};
</script>

<style>
canvas {
  max-width: 100%;
}
.styled-audio {
  width: 100%;
  margin-top: 10px;
  border-radius: 5px;
}
</style>
