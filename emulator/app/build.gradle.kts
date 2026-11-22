import javax.inject.Inject

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

abstract class PreparePipelineAssets @Inject constructor(private val files: FileSystemOperations) : DefaultTask() {
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val demoPack: DirectoryProperty

    @get:OutputDirectory
    abstract val output: DirectoryProperty

    @TaskAction
    fun prepare() {
        files.sync {
            from(demoPack) { include("config.json", "head.json", "models/**") }
            into(output)
        }
    }
}

android {
    namespace = "com.earshot.app"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.earshot.app"
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = true
                packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        noCompress += "tflite"
    }
}

androidComponents.onVariants { variant ->
    val prepare = tasks.register<PreparePipelineAssets>("prepare${variant.name.replaceFirstChar(Char::uppercaseChar)}PipelineAssets") {
        demoPack.set(rootProject.layout.projectDirectory.dir("../demo_pack"))
    }
    variant.sources.assets?.addGeneratedSourceDirectory(prepare, PreparePipelineAssets::output)
}

dependencies {
    implementation(libs.litert)
    implementation(libs.vosk) { artifact { type = "aar" } }
    implementation(libs.jna) { artifact { type = "aar" } }
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
